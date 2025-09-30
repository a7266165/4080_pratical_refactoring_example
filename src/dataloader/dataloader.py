# src/dataloader/dataloader.py
"""簡化版資料載入管線"""
import logging
from typing import Dict, List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler

from config.path_config import DATA_PATHS, get_demo_path
from src.dataloader.featureloader import FeatureLoader
from src.dataloader.selector.data_selector import DataSelector
from src.utils.utils import parse_subject_id

logger = logging.getLogger(__name__)


class DataLoader:
    """資料載入器 - 整合篩選和載入功能"""
    
    def __init__(
        self,
        embedding_models: List[str],
        feature_types: List[str],
        use_all_visits: bool = False,
        age_matching: bool = True,
        cdr_thresholds: Optional[List[float]] = None
    ):
        """
        Args:
            embedding_models: 嵌入模型列表
            feature_types: 特徵類型列表 (difference, average, relative)
            use_all_visits: 是否使用所有訪視
            age_matching: 是否進行年齡配對
            cdr_thresholds: CDR篩選閾值列表
        """
        self.embedding_models = embedding_models
        self.feature_types = feature_types
        self.use_all_visits = use_all_visits
        self.age_matching = age_matching
        self.cdr_thresholds = cdr_thresholds or []
        
        # 路徑設定
        self.data_path = str(DATA_PATHS["features"])
        self.demo_paths = {
            "p_csv": get_demo_path("p_csv"),
            "acs_csv": get_demo_path("acs_csv"),
            "nad_csv": get_demo_path("nad_csv")
        }
        
        # 初始化資料選擇器
        self.selectors = self._initialize_selectors()
    
    def _initialize_selectors(self) -> Dict[str, DataSelector]:
        """初始化資料選擇器"""
        selectors = {}
        
        if self.cdr_thresholds:
            # 為每個CDR閾值建立選擇器
            for threshold in self.cdr_thresholds:
                logger.info(f"初始化 CDR>{threshold} 的選擇器...")
                
                selector = DataSelector(
                    demo_paths=self.demo_paths,
                    age_matching=self.age_matching,
                    cdr_filter=True,
                    cdr_threshold=threshold
                )
                selector.build_selection()
                
                selectors[f"cdr_{threshold}"] = selector
                
                logger.info(
                    f"CDR>{threshold}: "
                    f"P={len(selector.allowed_ids.get('P', []))}人, "
                    f"ACS={len(selector.allowed_ids.get('ACS', []))}人, "
                    f"NAD={len(selector.allowed_ids.get('NAD', []))}人"
                )
        else:
            # 沒有CDR篩選
            logger.info("初始化標準選擇器...")
            
            selector = DataSelector(
                demo_paths=self.demo_paths,
                age_matching=self.age_matching,
                cdr_filter=False
            )
            selector.build_selection()
            
            selectors["standard"] = selector
        
        return selectors
    
    def load(self) -> Dict:
        """載入所有配置的資料集
        
        Returns:
            資料集字典，key為配置名稱
        """
        logger.info("開始載入特徵資料...")
        datasets = {}
        
        # 對每個選擇器載入資料
        for selector_key, selector in self.selectors.items():
            logger.info(f"\n處理 {selector_key} 資料集...")
            
            # 建立特徵載入器
            loader = FeatureLoader(
                self.data_path,
                allowed_subjects=selector.allowed_ids,
                use_all_visits=self.use_all_visits
            )
            
            # 掃描個案
            subjects = loader.scan_subjects()
            logger.info(f"  掃描到 {len(subjects)} 個個案")
            
            # 對每個嵌入模型和特徵類型組合
            for embedding_model in self.embedding_models:
                for feature_type in self.feature_types:
                    # 建立資料集key
                    if selector_key == "standard":
                        dataset_key = f"{embedding_model}_{feature_type}"
                    else:
                        dataset_key = f"{embedding_model}_{feature_type}_{selector_key}"
                    
                    # 載入特徵
                    feature_data = loader.load_features(
                        subjects,
                        embedding_model=embedding_model,
                        feature_type=feature_type
                    )
                    
                    if not feature_data:
                        logger.warning(f"  {dataset_key}: 無資料")
                        continue
                    
                    # 準備訓練資料
                    X, y, subject_ids = self._prepare_training_data(
                        feature_data,
                        selector.lookup_table
                    )
                    
                    if len(X) == 0:
                        continue
                    
                    # 建立資料集
                    datasets[dataset_key] = {
                        "X": X,
                        "y": y,
                        "subject_ids": subject_ids,
                        "metadata": {
                            "embedding_model": embedding_model,
                            "feature_type": feature_type,
                            "use_all_visits": self.use_all_visits,
                            "age_matching": self.age_matching,
                            "cdr_threshold": selector.cdr_threshold if hasattr(selector, 'cdr_threshold') else None,
                            "n_samples": len(X),
                            "n_features": X.shape[1],
                            "n_health": np.sum(y == 0),
                            "n_patient": np.sum(y == 1)
                        }
                    }
                    
                    logger.info(
                        f"  {dataset_key}: {len(X)} 樣本, "
                        f"{X.shape[1]} 特徵"
                    )
        
        logger.info(f"\n總共載入 {len(datasets)} 個資料集配置")
        return datasets
    
    def _prepare_training_data(self, feature_data, lookup_table):
        """準備訓練資料
        
        Returns:
            X: 特徵矩陣
            y: 標籤向量
            subject_ids: 受試者ID列表
        """
        X_list, y_list, subject_ids = [], [], []
        
        for fd in feature_data:
            # 提取特徵
            features = list(fd.features.values())[0]
            if features is None:
                continue
            
            X_list.append(features)
            y_list.append(fd.subject_info.label)
            subject_ids.append(fd.subject_info.subject_id)
        
        if not X_list:
            return np.array([]), np.array([]), []
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 整合人口學特徵（如果有提供查詢表）
        if lookup_table:
            X = self._add_demographics(X, subject_ids, lookup_table)
        
        return X, y, subject_ids
    
    def _add_demographics(self, X, subject_ids, lookup_table):
        """添加人口學特徵
        
        Args:
            X: 原始特徵矩陣
            subject_ids: 受試者ID列表
            lookup_table: 人口學查詢表
        
        Returns:
            結合人口學特徵的特徵矩陣
        """
        # 提取年齡和性別
        demo_features = []
        
        for sid in subject_ids:
            # 嘗試多種ID格式查詢
            meta = lookup_table.get(sid)
            if meta is None:
                base_id, _ = parse_subject_id(sid)
                meta = lookup_table.get(base_id)
            
            if meta:
                age = meta.get("Age", np.nan)
                sex = meta.get("Sex", np.nan)
            else:
                age = np.nan
                sex = np.nan
            
            demo_features.append([age, sex])
        
        demo_array = np.array(demo_features)
        
        # 填補缺失值
        age_mean = np.nanmean(demo_array[:, 0])
        sex_mode = lookup_table.get("_SEX_MODE_", 0.5)
        
        if np.isnan(age_mean):
            age_mean = 70  # 預設年齡
        
        demo_array[np.isnan(demo_array[:, 0]), 0] = age_mean
        demo_array[np.isnan(demo_array[:, 1]), 1] = sex_mode
        
        # 標準化並結合
        scaler_X = StandardScaler()
        scaler_demo = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        demo_scaled = scaler_demo.fit_transform(demo_array)
        
        # 結合特徵
        X_combined = np.hstack([X_scaled, demo_scaled])
        
        logger.debug(f"添加人口學特徵: {X.shape} -> {X_combined.shape}")
        
        return X_combined