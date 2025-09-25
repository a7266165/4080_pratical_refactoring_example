import os
os.environ["OMP_PROC_BIND"] = "false"    # 不要求綁定執行緒到特定 place
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS 才會用到，跨平台放著無妨
os.environ.setdefault("KMP_BLOCKTIME", "1")
os.environ.pop("KMP_AFFINITY", None)
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# 限縮 OpenCV 與 PyTorch 的執行緒數
try:
    cv2.setNumThreads(1)      # OpenCV 不要開到多執行緒
except Exception:
    pass

try:
    torch.set_num_threads(1)          # 計算執行緒（intra-op）
    torch.set_num_interop_threads(1)  # 運算子間併行（inter-op）
except Exception:
    pass

import time
THROTTLE_SLEEP = 0.001  # 每處理一張圖後

# 主要使用 DeepFace
from deepface import DeepFace
# face_recognition 用於 Dlib
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("face_recognition 未安裝，將使用 DeepFace 的 Dlib")

# ===== TopoFR 設定 =====
# 請根據您的實際路徑修改
TOPOFR_PATH = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\code\DeepLearning\TopoFR"  # TopoFR 資料夾路徑
TOPOFR_MODEL = "Glint360K_R100_TopoFR_9760.pt"  # 使用的模型檔案名稱

# 將 TopoFR 路徑加入系統路徑
if os.path.exists(TOPOFR_PATH):
    sys.path.insert(0, TOPOFR_PATH)
    print(f"✓ TopoFR 路徑已加入: {TOPOFR_PATH}")

class RobustFaceEmbeddingExtractor:
    """穩定版本的人臉特徵提取器（含 TopoFR）"""
    
    def __init__(self):
        print("="*60)
        print("初始化人臉特徵提取器")
        print("="*60)
        
        # 初始化 TopoFR
        self.topofr_model = None
        self.topofr_available = False
        
        # 檢查可用的模型
        self.check_available_models()
        
        # TODO: 待Debug
        # 嘗試載入 TopoFR
        self.init_topofr()
        
    def check_available_models(self):
        """檢查哪些模型可用"""
        self.available_models = {
            'VGG-Face': True,  # DeepFace 內建
            'ArcFace': True,   # 使用 DeepFace 版本
            'Dlib': True,      # DeepFace 內建
            'DeepID': True,    # DeepFace 內建
            'TopoFR': False    # 稍後檢查
        }
        
        print("模型狀態:")
        print("  ✓ VGG-Face: 可用 (DeepFace)")
        print("  ✓ ArcFace: 可用 (DeepFace)")
        print("  ✓ Dlib: 可用 (DeepFace)")
        print("  ✓ DeepID: 可用 (DeepFace)")
        # TopoFR 狀態會在 init_topofr 後更新
        
    def init_topofr(self):
        """初始化 TopoFR 模型"""
        try:
            # 檢查 TopoFR 路徑是否存在
            if not os.path.exists(TOPOFR_PATH):
                print(f"  ✗ TopoFR: 找不到 TopoFR 資料夾: {TOPOFR_PATH}")
                return
            
            # 檢查模型檔案
            model_path = os.path.join(TOPOFR_PATH, "model", TOPOFR_MODEL)
            if not os.path.exists(model_path):
                print(f"  ✗ TopoFR: 找不到模型檔案: {model_path}")
                return
            
            # 載入 TopoFR 模組
            from backbones import get_model
            
            # 判斷模型架構
            if "R50" in TOPOFR_MODEL:
                network = "r50"
                print("  載入 TopoFR R50 模型...")
            elif "R100" in TOPOFR_MODEL:
                network = "r100"
                print("  載入 TopoFR R100 模型...")
            elif "R200" in TOPOFR_MODEL:
                network = "r200"
                print("  載入 TopoFR R200 模型...")
            else:
                network = "r100"
                print("  載入 TopoFR R100 模型（預設）...")
            
            # 建立模型
            self.topofr_model = get_model(network, fp16=False)
            
            # 載入權重
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # device = torch.device('cpu')  # 強制使用 CPU，避免 CUDA 相關錯誤
            checkpoint = torch.load(model_path, map_location=device)
            
            # 處理 checkpoint 格式
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            
            # 載入權重
            self.topofr_model.load_state_dict(checkpoint, strict=False)
            self.topofr_model.to(device)
            self.topofr_model.eval()
            
            self.topofr_device = device
            self.topofr_available = True
            self.available_models['TopoFR'] = True
            
            print(f"  ✓ TopoFR: 成功載入 ({TOPOFR_MODEL})")
            print(f"    設備: {device}")
            
        except Exception as e:
            print(f"  ✗ TopoFR: 載入失敗 - {str(e)[:-1]}")
            self.topofr_available = False
        
        print()
        
    def extract_vggface(self, img_path: str) -> np.ndarray:
        """提取 VGG-Face 特徵"""
        try:
            result = DeepFace.represent(
                img_path=str(img_path),
                model_name='VGG-Face',
                enforce_detection=False,
                detector_backend='opencv',
                align=True
            )
            if result and len(result) > 0:
                return np.array(result[0]['embedding'])
        except Exception as e:
            print(f"    VGG-Face 錯誤: {str(e)[:50]}")
        return None
        
    def extract_arcface(self, img_path: str) -> np.ndarray:
        """提取 ArcFace 特徵 - 使用 DeepFace 版本"""
        try:
            # 使用 DeepFace 的 ArcFace（比 InsightFace 更穩定）
            result = DeepFace.represent(
                img_path=str(img_path),
                model_name='ArcFace',
                enforce_detection=False,  # 重要：不強制偵測臉部
                detector_backend='opencv',
                align=True
            )
            if result and len(result) > 0:
                return np.array(result[0]['embedding'])
        except Exception as e:
            print(f"    ArcFace 錯誤: {str(e)[:50]}")
        return None
        
    def extract_dlib(self, img_path: str) -> np.ndarray:
        """提取 Dlib 特徵"""
        # 優先使用 face_recognition（如果有安裝）
        if FACE_RECOGNITION_AVAILABLE:
            try:
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0:
                    return face_encodings[0]
            except:
                pass
        
        # 否則使用 DeepFace 的 Dlib
        try:
            result = DeepFace.represent(
                img_path=str(img_path),
                model_name='Dlib',
                enforce_detection=False,
                detector_backend='opencv',
                align=True
            )
            if result and len(result) > 0:
                return np.array(result[0]['embedding'])
        except Exception as e:
            print(f"    Dlib 錯誤: {str(e)[:50]}")
        return None
        
    def extract_deepid(self, img_path: str) -> np.ndarray:
        """提取 DeepID 特徵"""
        try:
            result = DeepFace.represent(
                img_path=str(img_path),
                model_name='DeepID',
                enforce_detection=False,
                detector_backend='opencv',
                align=True
            )
            if result and len(result) > 0:
                return np.array(result[0]['embedding'])
        except Exception as e:
            print(f"    DeepID 錯誤: {str(e)[:50]}")
        return None
        
    def extract_topofr(self, img_path: str) -> np.ndarray:
        """提取 TopoFR 特徵"""
        if not self.topofr_available or self.topofr_model is None:
            return None
        
        try:
            # 讀取圖片
            img = cv2.imread(img_path)
            if img is None:
                print(f"    TopoFR: 無法讀取圖片")
                return None
            
            # 預處理：TopoFR 使用 112x112 輸入
            img = cv2.resize(img, (112, 112))
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 轉換為 tensor
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            img = torch.from_numpy(img).unsqueeze(0).float()
            
            # 正規化到 [-1, 1]
            img = img.div(255).sub(0.5).div(0.5)
            img = img.to(self.topofr_device)
            
            # 提取特徵
            with torch.no_grad():
                embedding = self.topofr_model(img)
                # 正規化特徵向量（TopoFR 通常會做 L2 正規化）
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embedding = embedding.cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            print(f"    TopoFR 錯誤: {str(e)[:50]}")
            return None
        
    def extract_all_embeddings(self, img_path: str) -> dict:
        """提取所有模型的特徵"""
        img_name = Path(img_path).name
        print(f"\n處理圖片: {img_name}")
        
        embeddings = {}
        
        # 1. VGG-Face
        print("  提取 VGG-Face...", end=" ")
        vgg_emb = self.extract_vggface(img_path)
        embeddings['vggface'] = vgg_emb.tolist() if vgg_emb is not None else None
        print("✓" if vgg_emb is not None else "✗")
        
        # 2. ArcFace (使用 DeepFace 版本，更穩定)
        print("  提取 ArcFace...", end=" ")
        arc_emb = self.extract_arcface(img_path)
        embeddings['arcface'] = arc_emb.tolist() if arc_emb is not None else None
        print("✓" if arc_emb is not None else "✗")
        
        # 3. Dlib
        print("  提取 Dlib...", end=" ")
        dlib_emb = self.extract_dlib(img_path)
        embeddings['dlib'] = dlib_emb.tolist() if dlib_emb is not None else None
        print("✓" if dlib_emb is not None else "✗")
        
        # 4. DeepID
        print("  提取 DeepID...", end=" ")
        deepid_emb = self.extract_deepid(img_path)
        embeddings['deepid'] = deepid_emb.tolist() if deepid_emb is not None else None
        print("✓" if deepid_emb is not None else "✗")
        
        # TODO: 待Debug
        # 5. TopoFR
        print("  提取 TopoFR...", end=" ")
        topofr_emb = self.extract_topofr(img_path)
        embeddings['topofr'] = topofr_emb.tolist() if topofr_emb is not None else None
        if self.topofr_available:
            print("✓" if topofr_emb is not None else "✗")
        else:
            print("✗ (未載入)")
        
        return embeddings

# === 新增：檢查既有向量檔是否完整的工具函式（不改動原有結構/註解） ===
def is_complete_embedding_file(json_path: Path, required_models=None) -> bool:
    """檢查 *_embeddings.json 是否包含指定模型且皆為非 None，且維度資訊齊全"""
    if required_models is None:
        required_models = ['vggface', 'arcface', 'dlib', 'deepid']
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        embeddings = data.get('embeddings', {})
        dims = data.get('embedding_dimensions', {})
        succ = data.get('extraction_successful', {})
        if not isinstance(embeddings, dict) or not isinstance(dims, dict):
            return False
        for m in required_models:
            if m not in embeddings or embeddings[m] is None:
                return False
            if dims.get(m) is None:
                return False
            # 若有長度可比對則檢查一致
            if isinstance(embeddings[m], list) and isinstance(dims[m], int):
                if len(embeddings[m]) != dims[m]:
                    return False
            # 如果檔案內有 extraction_successful，且為 False 視為不完整
            if succ and succ.get(m) is False:
                return False
        return True
    except Exception:
        return False

def process_folder_structure(input_folder: str, output_folder: str):
    """處理整個資料夾結構"""
    
    # 初始化提取器
    extractor = RobustFaceEmbeddingExtractor()
    
    # 建立路徑
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支援的圖片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 收集所有圖片
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"錯誤: 在 {input_folder} 中沒有找到圖片")
        return
    
    print(f"\n找到 {len(image_files)} 張圖片待處理")
    print("="*60)
    
    # 統計變數
    success_count = 0
    failed_files = []
    model_stats = {
        'vggface': 0,
        'arcface': 0,
        'dlib': 0,
        'deepid': 0,
        'topofr': 0 # TODO: 待Debug
    }
    # === 新增：計數略過數量 ===
    skipped_count = 0
    
    # 處理每張圖片
    for img_path in tqdm(image_files, desc="總進度", unit="張"):
        try:
            # 保持資料夾結構
            relative_path = img_path.relative_to(input_path)
            output_dir = output_path / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # === 新增：若既有完整向量檔則跳過 ===
            existing_json = output_dir / (img_path.stem + '_embeddings.json')
            if output_dir.exists() and existing_json.exists():
                # 以目前啟用的模型鍵作為完整性要求
                required_models = list(model_stats.keys())
                if is_complete_embedding_file(existing_json, required_models=required_models):
                    skipped_count += 1
                    # 使用 tqdm.write 避免干擾進度列
                    tqdm.write(f"跳過圖片: {img_path.name}（已存在完整向量檔）")
                    continue
            
            # 提取特徵
            embeddings = extractor.extract_all_embeddings(str(img_path))
            
            # 統計成功的模型
            for model_name, embedding in embeddings.items():
                if embedding is not None:
                    model_stats[model_name] += 1
            
            # 計算向量維度
            dimensions = {}
            for model_name, embedding in embeddings.items():
                if embedding is not None:
                    dimensions[model_name] = len(embedding)
                else:
                    dimensions[model_name] = None
            
            # 準備輸出資料
            result = {
                'source_image': str(relative_path),
                'image_name': img_path.name,
                'embeddings': embeddings,
                'embedding_dimensions': dimensions,
                'extraction_successful': {
                    model: (emb is not None) 
                    for model, emb in embeddings.items()
                }
            }
            
            # 儲存 JSON
            output_file = output_dir / (img_path.stem + '_embeddings.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n錯誤處理 {img_path.name}: {str(e)[:100]}")
            failed_files.append(str(relative_path))
            
        time.sleep(THROTTLE_SLEEP)
    
    # 顯示統計結果
    print("\n" + "="*60)
    print("處理完成!")
    print(f"成功處理: {success_count}/{len(image_files)} 張圖片")
    
    if failed_files:
        print(f"\n失敗: {len(failed_files)} 張")
        for f in failed_files[:5]:
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... 還有 {len(failed_files)-5} 張")
    
    # 顯示各模型成功率
    print("\n各模型提取成功率:")
    for model_name, count in model_stats.items():
        pct = (count / len(image_files) * 100) if len(image_files) > 0 else 0
        status = "✓" if count > 0 else "✗"
        if model_name == 'topofr':
            if extractor.topofr_available:
                print(f"  {status} {model_name:10s}: {count:4d}/{len(image_files)} ({pct:5.1f}%) - [TopoFR {TOPOFR_MODEL}]")
            else:
                print(f"  {status} {model_name:10s}: {count:4d}/{len(image_files)} ({pct:5.1f}%) - [未載入]")
        else:
            print(f"  {status} {model_name:10s}: {count:4d}/{len(image_files)} ({pct:5.1f}%)")
    
    print(f"\n輸出資料夾: {output_folder}")
    # === 新增：輸出略過統計 ===
    print(f"跳過（已存在完整向量）: {skipped_count} 張")
    
    # 建立統計檔案
    create_summary_file(output_folder, model_stats, len(image_files))

def create_summary_file(output_folder: str, model_stats: dict, total_images: int):
    """建立統計摘要檔案"""
    output_path = Path(output_folder)
    
    # 計算向量維度（從第一個成功的檔案取得）
    dimensions = {}
    json_files = list(output_path.rglob('*_embeddings.json'))
    if json_files:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'embedding_dimensions' in data:
                dimensions = data['embedding_dimensions']
    
    # 建立摘要
    summary = {
        'total_images_processed': total_images,
        'model_success_count': model_stats,
        'model_success_rate': {
            model: f"{(count/total_images*100):.1f}%" 
            for model, count in model_stats.items()
        },
        'embedding_dimensions': dimensions,
        'notes': {
            'VGG-Face': '2622 維特徵向量',
            'ArcFace': '512 維特徵向量',
            'Dlib': '128 維特徵向量', 
            'DeepID': '160 維特徵向量',
            'TopoFR': f'512 維特徵向量 (使用 {TOPOFR_MODEL})' # TODO: 待Debug
        },
        # TODO: 待Debug
        'topofr_info': {
            'model_used': TOPOFR_MODEL,
            'path': TOPOFR_PATH,
            'status': 'loaded' if os.path.exists(os.path.join(TOPOFR_PATH, "model", TOPOFR_MODEL)) else 'not found'
        }
    }
    
    # 儲存摘要
    summary_file = output_path / 'extraction_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n統計摘要已儲存: {summary_file}")

def main():
    """主程式"""
    
    # ========== 請修改這些路徑 ==========
    INPUT_FOLDER = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\_pics\3_histogram_matched\datung\health\NAD"
    OUTPUT_FOLDER = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\_pics\4_pics_to_vector\datung\health\NAD_demo"
    
    # =======================================
    
    # 檢查輸入資料夾
    if not os.path.exists(INPUT_FOLDER):
        print("錯誤: 找不到輸入資料夾!")
        print(f"請確認路徑: {INPUT_FOLDER}")
        print("\n請修改程式中 main() 函數的 INPUT_FOLDER 變數")
        return
    
    print("人臉特徵向量提取程式")
    print("="*60)
    print(f"輸入資料夾: {INPUT_FOLDER}")
    print(f"輸出資料夾: {OUTPUT_FOLDER}")
    # TODO: 待Debug
    print(f"TopoFR 路徑: {TOPOFR_PATH}")
    print(f"TopoFR 模型: {TOPOFR_MODEL}")
    
    # 開始處理
    process_folder_structure(INPUT_FOLDER, OUTPUT_FOLDER)
    
    print("\n程式執行完成!")

if __name__ == "__main__":
    main()