# src/dataloader/dataloader.py
"""改進版特徵載入管線"""
import logging
from typing import Dict, List, Optional, Set
import numpy as np
from config.path_config import DATA_PATHS, get_demo_path
from src.dataloader.featureloader import FeatureLoader
from src.dataloader.selector.demographics import DemographicsLoader
from src.dataloader.selector.age_balancing import DataBalancer
from src.dataloader.selector.subject_filter import SubjectFilter
from src.utils.id_parser import parse_subject_id

logger = logging.getLogger(__name__)

class DataLoader:
    """改進版特徵載入器
    
    改進重點：
    1. 先根據篩選條件建立允許清單
    2. 只載入允許清單中的資料
    3. 避免載入不需要的資料
    4. 支援多個CDR閾值設定
    """
    
    def __init__(
        self,
        embedding_models: List[str],
        feature_types: List[str], 
        use_all_visits: bool = False,
        age_matching: bool = True,
        cdr_thresholds: Optional[List[float]] = None
    ):
        """
        初始化 DataLoader
        
        Args:
            embedding_models: 要使用的嵌入模型列表
            feature_types: 特徵類型列表 (difference, average, relative)
            use_all_visits: 是否使用所有訪視
            age_matching: 是否進行年齡配對
            cdr_thresholds: CDR篩選閾值列表，例如 [0.5, 1.0, 2.0]
                          空列表或None表示不進行CDR篩選
        """
        self.embedding_models = embedding_models
        self.feature_types = feature_types
        self.use_all_visits = use_all_visits
        self.age_matching = age_matching
        self.cdr_thresholds = cdr_thresholds or []
        
        self.data_path = str(DATA_PATHS["features"])
        self.demo_paths = self._get_demo_paths()
        
        # 為每個CDR閾值建立篩選器（如果沒有CDR閾值，則建立一個無CDR篩選的）
        self.subject_filters = self._initialize_filters()
        
    def _get_demo_paths(self) -> Dict:
        """從配置檔讀取人口學資料路徑"""
        return {
            'p_csv': get_demo_path('p_csv'),
            'acs_csv': get_demo_path('acs_csv'),
            'nad_csv': get_demo_path('nad_csv')
        }
        
    def _initialize_filters(self) -> Dict[str, SubjectFilter]:
        """初始化篩選器，為每個CDR閾值建立一個篩選器"""
        filters = {}
        
        if self.cdr_thresholds:
            # 為每個CDR閾值建立篩選器
            for threshold in self.cdr_thresholds:
                logger.info(f"初始化CDR>{threshold}的篩選器...")
                
                subject_filter = SubjectFilter(
                    demo_paths=self.demo_paths,
                    age_matching=self.age_matching,
                    cdr_filter=True,
                    cdr_threshold=threshold,
                    use_all_visits=self.use_all_visits
                )
                
                subject_filter.build_allowed_list()
                
                filter_key = f"cdr_{threshold}"
                filters[filter_key] = subject_filter
                
                logger.info(f"CDR>{threshold} 篩選器完成: "
                           f"P={len(subject_filter.allowed_ids.get('P', []))}人, "
                           f"ACS={len(subject_filter.allowed_ids.get('ACS', []))}人, "
                           f"NAD={len(subject_filter.allowed_ids.get('NAD', []))}人")
        else:
            # 沒有CDR篩選，建立單一篩選器
            logger.info("初始化標準篩選器（無CDR篩選）...")
            
            subject_filter = SubjectFilter(
                demo_paths=self.demo_paths,
                age_matching=self.age_matching,
                cdr_filter=False,
                cdr_threshold=None,
                use_all_visits=self.use_all_visits
            )
            
            subject_filter.build_allowed_list()
            
            filters["standard"] = subject_filter
            
            logger.info(f"標準篩選器完成: "
                       f"P={len(subject_filter.allowed_ids.get('P', []))}人, "
                       f"ACS={len(subject_filter.allowed_ids.get('ACS', []))}人, "
                       f"NAD={len(subject_filter.allowed_ids.get('NAD', []))}人")
        
        return filters
    
    def load(self) -> Dict:
        """載入特徵並建立訓練資料集（只載入篩選後的資料）"""
        logger.info("開始載入特徵資料（僅載入篩選後的個案）")
        
        datasets = {}
        
        # 對每個篩選器建立資料集
        for filter_key, subject_filter in self.subject_filters.items():
            logger.info(f"處理 {filter_key} 資料集...")
            
            # 使用篩選後的清單載入特徵
            featureloader = FeatureLoader(
                self.data_path,
                allowed_subjects=subject_filter.allowed_ids,
                use_all_visits=self.use_all_visits
            )
            
            # 準備所有組合的資料集
            for embedding_model in self.embedding_models:
                for feature_type in self.feature_types:
                    # 建立包含CDR資訊的key
                    if filter_key == "standard":
                        dataset_key = f"{embedding_model}_{feature_type}"
                    else:
                        dataset_key = f"{embedding_model}_{feature_type}_{filter_key}"
                    
                    # 只載入允許清單中的特徵
                    dataset = self._create_dataset(
                        featureloader,
                        subject_filter,
                        embedding_model,
                        feature_type,
                        filter_key
                    )
                    
                    if dataset:
                        datasets[dataset_key] = dataset
                        logger.info(f"  {dataset_key}: 載入 {len(dataset['X'])} 筆資料")
                    
        return datasets
    
    def _create_dataset(
        self,
        loader: FeatureLoader,
        subject_filter: SubjectFilter,
        embedding_model: str,
        feature_type: str,
        filter_key: str
    ) -> Optional[Dict]:
        """建立單一配置的訓練資料集"""
        
        # 只掃描允許清單中的個案
        subjects = loader.scan_allowed_subjects()
        
        # 載入特徵（已經是篩選後的）
        feature_data = loader.load_features(
            subjects,
            embedding_model=embedding_model,
            feature_type=feature_type
        )
        
        # 準備資料
        X_list, y_list, subject_ids = [], [], []
        
        for fd in feature_data:
            features = list(fd.features.values())[0]
            if features is not None:
                X_list.append(features)
                y_list.append(fd.subject_info.label)
                subject_ids.append(fd.subject_info.subject_id)
        
        if not X_list:
            return None
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 整合人口學特徵
        if subject_filter.demographics_lookup:
            X = self._integrate_demographics(
                X, subject_ids, 
                subject_filter.demographics_lookup
            )
        
        # 解析CDR閾值資訊
        cdr_info = None
        if filter_key.startswith("cdr_"):
            cdr_info = float(filter_key.replace("cdr_", ""))
        
        return {
            'X': X,
            'y': y,
            'subject_ids': subject_ids,
            'metadata': {
                'embedding_model': embedding_model,
                'feature_type': feature_type,
                'use_all_visits': self.use_all_visits,
                'age_matching': self.age_matching,
                'cdr_threshold': cdr_info,
                'filter_summary': subject_filter.get_summary()
            }
        }
    
    def _integrate_demographics(
        self,
        X: np.ndarray,
        subject_ids: List[str],
        lookup: Dict
    ) -> np.ndarray:
        """整合人口學特徵"""
        demo_features = []
        
        for sid in subject_ids:
            meta = lookup.get(sid)
            if meta is None:
                base_id, _ = parse_subject_id(sid)
                meta = lookup.get(base_id)
                
            if meta:
                age = meta.get('Age', np.nan)
                sex = meta.get('Sex', np.nan)
            else:
                age = np.nan
                sex = np.nan
                
            demo_features.append([age, sex])
        
        demo_array = np.array(demo_features)
        
        # 填補缺失值
        age_mean = np.nanmean(demo_array[:, 0])
        sex_mode = lookup.get("_SEX_MODE_", 0.5)
        
        if np.isnan(age_mean):
            age_mean = 70
            
        demo_array[np.isnan(demo_array[:, 0]), 0] = age_mean
        demo_array[np.isnan(demo_array[:, 1]), 1] = sex_mode
        
        # 標準化並結合
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_demo = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        demo_scaled = scaler_demo.fit_transform(demo_array)
        
        X_combined = np.hstack([X_scaled, demo_scaled])
        
        logger.debug(f"添加人口學特徵: {X.shape} -> {X_combined.shape}")
        return X_combined