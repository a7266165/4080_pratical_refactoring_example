# src/extractor.py
"""特徵提取模組"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import logging
from tqdm import tqdm

import torch
import cv2
from deepface import DeepFace

os.environ["OMP_PROC_BIND"] = "false"
os.environ.setdefault("KMP_BLOCKTIME", "1")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """特徵提取器：多模型臉部特徵提取與差異計算"""
    
    def __init__(self, models: List[str] = None):
        """
        Args:
            models: 要使用的模型列表（會自動加入 topofr 如果可用）
        """
        self.models = models
        self._init_topofr()
    
    def _init_topofr(self):
        """初始化 TopoFR 模型"""
        try:
            from config.path_config import TOPOFR_PATH
            
            TOPOFR_MODEL = "Glint360K_R100_TopoFR_9760.pt"

            model_path = Path(TOPOFR_PATH) / "model" / TOPOFR_MODEL
            if not model_path.exists():
                logger.warning(f"無法從 {model_path} 載入 TopoFR")
                return
            
            # 加入系統路徑並載入模組
            sys.path.insert(0, str(TOPOFR_PATH))
            from backbones import get_model
            
            # 從檔名判斷網路架構
            network = "r100"  # 預設
            for net in ["r50", "r100", "r200"]:
                if net.upper() in TOPOFR_MODEL.upper():
                    network = net
                    break
            
            # 載入模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.topofr_model = get_model(network, fp16=False)
            
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict):
                checkpoint = checkpoint.get('state_dict', checkpoint)
            
            self.topofr_model.load_state_dict(checkpoint, strict=False)
            self.topofr_model.to(device).eval()
            self.topofr_device = device
            
            logger.info(f"TopoFR ({network}) 載入成功，設備: {device}")
                
        except Exception as e:
            logger.error(f"TopoFR 載入失敗: {e}")
    
    # ========== 提取相片特徵向量 ==========
    def extract_embeddings(self, image_path: str) -> Dict[str, Optional[np.ndarray]]:
        """提取所有模型的特徵向量"""
        if not self.models:
            logger.warning("沒有指定任何模型")
            return {}
        
        embeddings = {}
        for model_name in self.models:
            try:
                if model_name == 'topofr':
                    embeddings[model_name] = self._extract_topofr(image_path)
                else:
                    embeddings[model_name] = self._extract_deepface(image_path, model_name)
            except Exception as e:
                logger.debug(f"{model_name} 提取失敗: {e}")
                embeddings[model_name] = None
        
        return embeddings
    
    def _extract_deepface(self, img_path: str, model_name: str) -> Optional[np.ndarray]:
        """使用 DeepFace 提取特徵"""
        name_map = {
            'vggface': 'VGG-Face',
            'arcface': 'ArcFace', 
            'dlib': 'Dlib',
            'deepid': 'DeepID'
        }
        
        result = DeepFace.represent(
            img_path=str(img_path),
            model_name=name_map.get(model_name, model_name),
            enforce_detection=False,
            detector_backend='opencv',
            align=True
        )
        
        assert result and 'embedding' in result[0], f"{model_name} 無法提取特徵"

        return np.array(result[0]['embedding'])
                
    def _extract_topofr(self, img_path: str) -> Optional[np.ndarray]:
        """提取 TopoFR 特徵"""
        assert hasattr(self, 'topofr_model'), "TopoFR 模型未載入"

        img = cv2.imread(img_path)
        assert img is not None, f"無法讀取圖片: {img_path}"

        # 預處理
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.div(255).sub(0.5).div(0.5).to(self.topofr_device)
        
        # 提取特徵
        with torch.no_grad():
            embedding = self.topofr_model(img)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy()[0]
            
    # ========== 計算不對稱性 ==========
    def calculate_asymmetry(
        self,
        left_embeddings: Dict[str, Optional[np.ndarray]],
        right_embeddings: Dict[str, Optional[np.ndarray]]
    ) -> Dict:
        """計算左右臉不對稱性指標"""
        result = {
            "embedding_differences": {},
            "embedding_averages": {},
            "relative_differences": {},
            "extraction_successful": {}
        }
        
        for model_name in (self.models or []):
            left = left_embeddings.get(model_name)
            right = right_embeddings.get(model_name)
            
            if left is None or right is None:
                result["extraction_successful"][model_name] = False
                continue
            
            result["extraction_successful"][model_name] = True
            
            # 基本計算
            diff = left - right
            avg = (left + right) / 2
            
            # 相對差異
            abs_diff = np.abs(diff)
            abs_sum = np.abs(left + right)
            relative_diff = np.where(abs_sum > 0, abs_diff / abs_sum, 0.0)
            
            # 儲存結果
            result["embedding_differences"][model_name] = diff.tolist()
            result["embedding_averages"][model_name] = avg.tolist()
            result["relative_differences"][model_name] = relative_diff.tolist()
        
        # 簡化的整體分數
        if result["relative_differences"]:
            all_diffs = []
            for diffs in result["relative_differences"].values():
                all_diffs.extend(diffs)
            result["overall_asymmetry"] = float(np.mean(all_diffs)) if all_diffs else 0.0
        
        return result
    
    # ========== 資料夾處理函數 ==========
    def process_folder(self, input_dir: str, output_dir: str) -> int:
        """處理資料夾中的影像對"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        processed = 0
        
        left_files = list(input_path.rglob("*_Lmirror*.png"))
        logger.info(f"找到 {len(left_files)} 個左臉檔案")

        # 只找左臉檔案，自動推導右臉
        for left_file in tqdm(left_files, desc="處理影像對"):
            right_file = Path(str(left_file).replace("_Lmirror", "_Rmirror"))
            
            if not right_file.exists():
                continue
            
            # 建立輸出路徑
            rel_path = left_file.parent.relative_to(input_path)
            out_dir = output_path / rel_path
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 輸出檔名
            base = left_file.stem.replace("_Lmirror_claheL", "")
            out_file = out_dir / f"{base}_LR_difference.json"
            
            try:
                self._process_image_pair(str(left_file), str(right_file), str(out_file))
                processed += 1
            except Exception as e:
                logger.warning(f"{left_file.name}: {e}")
        
        logger.info(f"處理完成: {processed} 對")
        return processed
    
    def _process_image_pair(
        self,
        left_image_path: str,
        right_image_path: str,
        output_path: str
    ) -> Dict:
        """處理一對左右臉影像"""
        # 提取特徵
        left_embeddings = self.extract_embeddings(left_image_path)
        right_embeddings = self.extract_embeddings(right_image_path)
        
        # 計算不對稱性
        result = self.calculate_asymmetry(left_embeddings, right_embeddings)
        
        # 儲存結果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result