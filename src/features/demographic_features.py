# src/features/demographic_features.py
"""人口學特徵整合模組"""
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DemographicFeatureConfig:
    """人口學特徵配置"""
    include_age: bool = True
    include_sex: bool = True
    weight: float = 1.0  # 人口學特徵的權重
    standardize: bool = True


class DemographicFeatureIntegrator:
    """人口學特徵整合器
    
    負責將年齡、性別等人口學特徵與影像特徵結合
    """
    
    def __init__(self, config: Optional[DemographicFeatureConfig] = None):
        self.config = config or DemographicFeatureConfig()
        self.feature_mean = None
        self.feature_std = None
        self.demo_mean = None
        self.demo_std = None
        
    def integrate_features(
        self,
        embedding_features: np.ndarray,
        subject_ids: List[str],
        demo_lookup: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
        """整合人口學特徵與嵌入特徵
        
        Args:
            embedding_features: 影像嵌入特徵矩陣 (n_samples, n_features)
            subject_ids: 受試者ID列表
            demo_lookup: 人口學查詢表
            
        Returns:
            整合後的特徵矩陣
        """
        if not self.config.include_age and not self.config.include_sex:
            return embedding_features
        
        # 提取人口學特徵
        demo_features = self._extract_demographic_features(subject_ids, demo_lookup)
        
        if demo_features is None:
            logger.warning("無法提取人口學特徵，返回原始特徵")
            return embedding_features
        
        # 標準化
        if self.config.standardize:
            embedding_z = self._standardize(embedding_features, fit=True, prefix="embed")
            demo_z = self._standardize(demo_features, fit=True, prefix="demo")
            
            # 應用權重並串接
            combined = np.hstack([embedding_z, self.config.weight * demo_z])
        else:
            # 直接串接
            combined = np.hstack([embedding_features, self.config.weight * demo_features])
        
        logger.info(f"特徵整合完成: {embedding_features.shape} + {demo_features.shape} -> {combined.shape}")
        
        return combined
    
    def _extract_demographic_features(
        self,
        subject_ids: List[str],
        demo_lookup: Dict[str, Dict[str, float]]
    ) -> Optional[np.ndarray]:
        """提取人口學特徵"""
        demo_list = []
        sex_mode = demo_lookup.get("_SEX_MODE_", np.nan)
        
        for subject_id in subject_ids:
            # 嘗試不同的ID格式查詢
            meta = demo_lookup.get(subject_id)
            if meta is None:
                # 嘗試base_id
                from src.utils.id_parser import parse_subject_id
                base_id, _ = parse_subject_id(subject_id)
                meta = demo_lookup.get(base_id)
            
            if meta is None:
                # 使用預設值
                age = np.nan
                sex = sex_mode
            else:
                age = meta.get("Age", np.nan)
                sex = meta.get("Sex", sex_mode if np.isnan(meta.get("Sex", np.nan)) else meta["Sex"])
            
            # 根據配置選擇特徵
            features = []
            if self.config.include_age:
                features.append(age)
            if self.config.include_sex:
                features.append(sex)
            
            demo_list.append(features)
        
        demo_arr = np.array(demo_list, dtype=np.float64)
        
        # 填補缺失值
        for col in range(demo_arr.shape[1]):
            col_data = demo_arr[:, col]
            if np.isnan(col_data).any():
                # 使用該欄位的平均值填補
                mean_val = np.nanmean(col_data)
                if np.isnan(mean_val):
                    mean_val = 0.0
                col_data[np.isnan(col_data)] = mean_val
                demo_arr[:, col] = col_data
        
        return demo_arr
    
    def _standardize(self, X: np.ndarray, fit: bool = True, prefix: str = "") -> np.ndarray:
        """標準化特徵"""
        if fit:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            
            # 儲存參數
            if prefix == "embed":
                self.feature_mean = mean
                self.feature_std = std
            elif prefix == "demo":
                self.demo_mean = mean
                self.demo_std = std
        else:
            # 使用已有參數
            if prefix == "embed":
                mean = self.feature_mean
                std = self.feature_std
            elif prefix == "demo":
                mean = self.demo_mean
                std = self.demo_std
            else:
                raise ValueError(f"Unknown prefix: {prefix}")
        
        return (X - mean) / std