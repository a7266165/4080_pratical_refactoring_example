# src/extractor.py
"""特徵提取模組"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import cv2
from deepface import DeepFace

os.environ["OMP_PROC_BIND"] = "false"
os.environ.setdefault("KMP_BLOCKTIME", "1")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """特徵提取器：多模型臉部特徵提取與差異計算"""
    
    def __init__(
        self,
        models: List[str] = None,
        use_topofr: bool = True  # 預設啟用
    ):
        """
        Args:
            models: 要使用的模型列表
            use_topofr: 是否使用 TopoFR
        """
        self.models = models or ['vggface', 'arcface', 'dlib', 'deepid']
        self.use_topofr = use_topofr
        
        # TopoFR 相關
        self.topofr_available = False
        self.topofr_model = None
        self.topofr_device = None
        
        # 從 path_config 取得 TopoFR 設定
        if use_topofr:
            from config.path_config import TOPOFR_PATH, TOPOFR_MODEL
            self.topofr_path = str(TOPOFR_PATH)
            self.topofr_model_name = TOPOFR_MODEL
            self._init_topofr()
        
        logger.info(f"初始化特徵提取器，模型: {self.models}")
        if self.topofr_available:
            logger.info(f"  TopoFR: 已載入 ({self.topofr_model_name})")
    
    def _init_topofr(self):
        """初始化 TopoFR 模型"""
        try:
            if not os.path.exists(self.topofr_path):
                logger.warning(f"TopoFR 路徑不存在: {self.topofr_path}")
                return
            
            # 檢查模型檔案
            model_path = os.path.join(self.topofr_path, "model", self.topofr_model_name)
            if not os.path.exists(model_path):
                logger.warning(f"TopoFR 模型檔案不存在: {model_path}")
                return
            
            # 將 TopoFR 路徑加入系統路徑
            sys.path.insert(0, self.topofr_path)
            
            # 載入 TopoFR 模組
            from backbones import get_model
            
            # 判斷模型架構
            if "R50" in self.topofr_model_name:
                network = "r50"
            elif "R100" in self.topofr_model_name:
                network = "r100"
            elif "R200" in self.topofr_model_name:
                network = "r200"
            else:
                network = "r100"
            
            logger.info(f"載入 TopoFR {network.upper()} 模型...")
            
            # 建立模型
            self.topofr_model = get_model(network, fp16=False)
            
            # 載入權重
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            
            self.topofr_model.load_state_dict(checkpoint, strict=False)
            self.topofr_model.to(device)
            self.topofr_model.eval()
            
            self.topofr_device = device
            self.topofr_available = True
            
            if 'topofr' not in self.models:
                self.models.append('topofr')
            
            logger.info(f"TopoFR 載入成功，設備: {device}")
            
        except Exception as e:
            logger.error(f"TopoFR 載入失敗: {str(e)}")
            self.topofr_available = False
    
    # ========== 提取相片特徵向量 ==========
    def extract_embeddings(self, image_path: str) -> Dict[str, Optional[np.ndarray]]:
        """提取所有模型的特徵向量
        
        Args:
            image_path: 影像路徑
            
        Returns:
            {model_name: embedding} 字典
        """
        embeddings = {}
        
        for model_name in self.models:
            if model_name == 'topofr' and self.topofr_available:
                embedding = self._extract_topofr(image_path)
            else:
                embedding = self._extract_deepface(image_path, model_name)
            
            embeddings[model_name] = embedding
        
        return embeddings
    
    def _extract_deepface(self, img_path: str, model_name: str) -> Optional[np.ndarray]:
        """使用 DeepFace 提取特徵"""
        try:
            # 根據模型調整名稱（DeepFace 格式）
            deepface_model_map = {
                'vggface': 'VGG-Face',
                'arcface': 'ArcFace',
                'dlib': 'Dlib',
                'deepid': 'DeepID'
            }
            
            model_name_deepface = deepface_model_map.get(model_name, model_name)
            
            result = DeepFace.represent(
                img_path=str(img_path),
                model_name=model_name_deepface,
                enforce_detection=False,
                detector_backend='opencv',
                align=True
            )
            
            if result and len(result) > 0:
                return np.array(result[0]['embedding'])
                
        except Exception as e:
            logger.debug(f"{model_name} 提取失敗: {str(e)[:50]}")
        
        return None
    
    def _extract_topofr(self, img_path: str) -> Optional[np.ndarray]:
        """提取 TopoFR 特徵"""
        if not self.topofr_available or self.topofr_model is None:
            return None
        
        try:
            # 讀取圖片
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # 預處理：TopoFR 使用 112x112 輸入
            img = cv2.resize(img, (112, 112))
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
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embedding = embedding.cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            logger.debug(f"TopoFR 提取失敗: {str(e)[:50]}")
            return None
    
    # ========== 計算不對稱性特徵 ==========
    def calculate_asymmetry(
        self,
        left_embeddings: Dict[str, Optional[np.ndarray]],
        right_embeddings: Dict[str, Optional[np.ndarray]]
    ) -> Dict:
        """計算左右臉不對稱性指標
        
        Args:
            left_embeddings: 左臉特徵
            right_embeddings: 右臉特徵
            
        Returns:
            包含差異、平均、相對差異等指標的字典
        """
        result = {
            "embedding_differences": {},
            "embedding_averages": {},
            "relative_differences": {},
            "embedding_dimensions": {},
            "extraction_successful": {},
            "statistics": {},
            "overall_asymmetry": {}
        }
        
        # 計算每個模型的差異
        for model_name in self.models:
            left_emb = left_embeddings.get(model_name)
            right_emb = right_embeddings.get(model_name)
            
            if left_emb is None or right_emb is None:
                result["extraction_successful"][model_name] = False
                continue
            
            result["extraction_successful"][model_name] = True
            result["embedding_dimensions"][model_name] = len(left_emb)
            
            # 差值 (左 - 右)
            diff = left_emb - right_emb
            result["embedding_differences"][model_name] = diff.tolist()
            
            # 平均值
            average = (left_emb + right_emb) / 2
            result["embedding_averages"][model_name] = average.tolist()
            
            # 相對差異（逐元素）
            abs_diff = np.abs(left_emb - right_emb)
            abs_sum = np.abs(left_emb + right_emb)
            relative_diff = np.where(abs_sum > 0, abs_diff / abs_sum, 0.0)
            result["relative_differences"][model_name] = relative_diff.tolist()
            
            # 統計資訊
            result["statistics"][model_name] = self._calculate_statistics(
                diff, average, relative_diff
            )
        
        # 整體不對稱度
        if result["relative_differences"]:
            all_relative_diffs = []
            for model_diffs in result["relative_differences"].values():
                all_relative_diffs.extend(model_diffs)
            
            if all_relative_diffs:
                all_relative_diffs = np.array(all_relative_diffs)
                result["overall_asymmetry"] = {
                    "mean_relative_difference": float(np.mean(all_relative_diffs)),
                    "std_relative_difference": float(np.std(all_relative_diffs)),
                    "max_relative_difference": float(np.max(all_relative_diffs)),
                    "min_relative_difference": float(np.min(all_relative_diffs)),
                    "median_relative_difference": float(np.median(all_relative_diffs)),
                    "percentile_75": float(np.percentile(all_relative_diffs, 75)),
                    "percentile_90": float(np.percentile(all_relative_diffs, 90)),
                    "percentile_95": float(np.percentile(all_relative_diffs, 95))
                }
        
        return result
    
    def _calculate_statistics(
        self,
        diff: np.ndarray,
        average: np.ndarray,
        relative_diff: np.ndarray
    ) -> Dict:
        """計算統計指標"""
        return {
            "difference": {
                "mean": float(np.mean(diff)),
                "std": float(np.std(diff)),
                "min": float(np.min(diff)),
                "max": float(np.max(diff)),
                "l2_norm": float(np.linalg.norm(diff))
            },
            "average": {
                "mean": float(np.mean(average)),
                "std": float(np.std(average)),
                "min": float(np.min(average)),
                "max": float(np.max(average)),
                "l2_norm": float(np.linalg.norm(average))
            },
            "relative_difference": {
                "mean": float(np.mean(relative_diff)),
                "std": float(np.std(relative_diff)),
                "min": float(np.min(relative_diff)),
                "max": float(np.max(relative_diff)),
                "median": float(np.median(relative_diff))
            },
            "asymmetry_score": float(np.linalg.norm(diff) / np.linalg.norm(average)) 
                if np.linalg.norm(average) > 0 else 0.0
        }
    
    # ========== 批次處理 ==========
    def process_image_pair(
        self,
        left_image_path: str,
        right_image_path: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """處理一對左右臉影像
        
        Args:
            left_image_path: 左臉影像路徑
            right_image_path: 右臉影像路徑  
            output_path: 輸出 JSON 路徑（可選）
            
        Returns:
            包含所有分析結果的字典
        """
        logger.debug(f"處理影像對: {Path(left_image_path).name} & {Path(right_image_path).name}")
        
        # 提取特徵
        left_embeddings = self.extract_embeddings(left_image_path)
        right_embeddings = self.extract_embeddings(right_image_path)
        
        # 計算不對稱性
        result = self.calculate_asymmetry(left_embeddings, right_embeddings)
        
        # 加入來源資訊
        result["source_images"] = {
            "left": str(left_image_path),
            "right": str(right_image_path)
        }
        result["image_names"] = {
            "left": Path(left_image_path).name,
            "right": Path(right_image_path).name
        }
        
        # 儲存結果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.debug(f"結果已儲存: {output_path}")
        
        return result
    
    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        pattern_left: str = "*_Lmirror*.png",
        pattern_right: str = "*_Rmirror*.png"
    ) -> Dict[str, List[Path]]:
        """批次處理資料夾中的影像對（支援巢狀結構）
        
        Args:
            input_dir: 輸入資料夾
            output_dir: 輸出資料夾
            pattern_left: 左臉檔名模式
            pattern_right: 右臉檔名模式
            
        Returns:
            處理結果路徑
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 預期的資料結構：health/ACS/ACS1-1/, health/NAD/NAD1-1/, patient/P1-2/
        # 需要建立對應的輸出結構
        
        # 遞迴尋找所有包含圖片的資料夾
        image_folders = []
        for root, dirs, files in os.walk(input_path):
            root_path = Path(root)
            # 檢查是否有左右臉配對
            left_files = list(root_path.glob(pattern_left))
            right_files = list(root_path.glob(pattern_right))
            
            if left_files and right_files:
                # 計算相對路徑
                rel_path = root_path.relative_to(input_path)
                image_folders.append((root_path, rel_path))
        
        logger.info(f"找到 {len(image_folders)} 個包含影像配對的資料夾")
        
        # 處理每個資料夾
        for folder_path, rel_path in image_folders:
            logger.info(f"處理資料夾: {rel_path}")
            
            # 建立對應的輸出資料夾結構
            output_subfolder = output_path / rel_path
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            # 找出配對的檔案
            pairs = self._find_paired_files(folder_path, pattern_left, pattern_right)
            logger.info(f"  找到 {len(pairs)} 對檔案")
            
            subfolder_results = []
            
            # 處理每對檔案
            for left_file, right_file in tqdm(pairs, desc=f"處理 {rel_path}"):
                try:
                    # 生成輸出檔名
                    base_name = left_file.name.replace("_Lmirror", "").replace("_claheL", "")
                    base_name = base_name.replace(".png", "")
                    output_filename = f"{base_name}_LR_difference.json"
                    output_filepath = output_subfolder / output_filename
                    
                    # 處理影像對
                    self.process_image_pair(
                        str(left_file),
                        str(right_file),
                        str(output_filepath)
                    )
                    
                    subfolder_results.append(output_filepath)
                    
                except Exception as e:
                    logger.warning(f"處理失敗: {left_file.name} & {right_file.name} - {e}")
                    continue
            
            results[str(rel_path)] = subfolder_results
        
        return results

    def _find_paired_files(
        self,
        folder: Path,
        pattern_left: str,
        pattern_right: str
    ) -> List[Tuple[Path, Path]]:
        """找出配對的左右臉檔案"""
        left_files = list(folder.glob(pattern_left))
        right_files = list(folder.glob(pattern_right))
        
        pairs = []
        
        for left_file in left_files:
            # 嘗試找出對應的右臉檔案
            base_name = left_file.name.replace("_Lmirror", "")
            
            for right_file in right_files:
                if base_name.replace("_claheL", "") == right_file.name.replace("_Rmirror", "").replace("_claheL", ""):
                    pairs.append((left_file, right_file))
                    break
        
        return pairs
    
    def create_summary_report(self, results_dir: str, output_path: str = None) -> Dict:
        """生成總結報告
        
        Args:
            results_dir: 包含 JSON 結果的資料夾
            output_path: 輸出報告路徑
            
        Returns:
            總結統計
        """
        results_path = Path(results_dir)
        all_results = []
        
        # 收集所有結果檔案
        for json_file in results_path.rglob("*_LR_difference.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data["file_path"] = str(json_file)
                data["folder"] = json_file.parent.name
                all_results.append(data)
        
        if not all_results:
            logger.warning("沒有找到任何結果檔案")
            return {}
        
        # 統計分析
        summary = {
            "total_files": len(all_results),
            "by_model": {},
            "high_asymmetry_cases": []
        }
        
        # 按模型統計
        for model_name in self.models:
            asymmetry_scores = []
            for result in all_results:
                if "statistics" in result and model_name in result["statistics"]:
                    score = result["statistics"][model_name].get("asymmetry_score", 0)
                    asymmetry_scores.append(score)
            
            if asymmetry_scores:
                summary["by_model"][model_name] = {
                    "mean_asymmetry": float(np.mean(asymmetry_scores)),
                    "std_asymmetry": float(np.std(asymmetry_scores)),
                    "max_asymmetry": float(np.max(asymmetry_scores)),
                    "min_asymmetry": float(np.min(asymmetry_scores)),
                    "median_asymmetry": float(np.median(asymmetry_scores))
                }
        
        # 找出高度不對稱的案例
        threshold = 0.1
        for result in all_results:
            if "overall_asymmetry" in result:
                if result["overall_asymmetry"]["mean_relative_difference"] > threshold:
                    summary["high_asymmetry_cases"].append({
                        "file": result.get("image_names", {}).get("left", ""),
                        "folder": result["folder"],
                        "asymmetry_score": result["overall_asymmetry"]["mean_relative_difference"]
                    })
        
        # 排序高度不對稱案例
        summary["high_asymmetry_cases"].sort(key=lambda x: x["asymmetry_score"], reverse=True)
        
        # 儲存報告
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"總結報告已儲存: {output_path}")
        
        return summary


# ========== 輔助函數 ==========
def batch_extract_features(
    input_dir: str,
    output_dir: str,
    models: List[str] = None,
    use_topofr: bool = False,
) -> Dict[str, List[Path]]:
    """批次提取特徵的便捷函數
    
    Args:
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        models: 使用的模型列表
        use_topofr: 是否使用 TopoFR
        topofr_path: TopoFR 路徑
        
    Returns:
        處理結果路徑
    """
    extractor = FeatureExtractor(
        models=models,
        use_topofr=use_topofr,
    )
    
    results = extractor.process_folder(input_dir, output_dir)
    
    # 生成總結報告
    summary_path = Path(output_dir) / "extraction_summary.json"
    extractor.create_summary_report(output_dir, summary_path)
    
    logger.info(f"特徵提取完成，共處理 {len(results)} 個資料夾")
    return results