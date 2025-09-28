# src/featureloader/featureloader.py
"""特徵載入管線"""
import logging
from typing import Dict, List, Optional
from pathlib import Path
import sys

# 暫時加入legacy_V2路徑以重用程式碼
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "legacy_V2"))

from src.data.loader import FeatureDataLoader
from src.data.demographics import DemographicsProcessor
from src.data.balancing import DataBalancer, BalancingConfig
from src.features.demographic_features import (
    DemographicFeatureIntegrator,
    DemographicFeatureConfig,
)

logger = logging.getLogger(__name__)


class FeatureLoader:
    """特徵載入器

    負責：
    1. 從檔案系統載入已計算的特徵
    2. 應用資料選擇策略（年齡平衡、CDR篩選等）
    3. 整合人口學特徵
    4. 輸出訓練資料集
    """

    def __init__(
        self,
        embedding_models: List[str],
        feature_types: List[str],
        use_all_visits: bool = False,
        age_matching: bool = True,
        cdr_filter: bool = False,
    ):
        self.embedding_models = embedding_models
        self.feature_types = feature_types
        self.use_all_visits = use_all_visits
        self.age_matching = age_matching
        self.cdr_filter = cdr_filter

        # 從設定檔讀取路徑
        self.data_path = self._get_data_path()
        self.demo_paths = self._get_demo_paths()

    def _get_data_path(self) -> str:
        # TODO: 從 config/path_config 讀取
        return r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature_V2\datung"

    def _get_demo_paths(self) -> Dict:
        # TODO: 從 config/path_config 讀取
        return {
            "p_csv": r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\p_merged.csv",
            "acs_csv": r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\ACS_merged_results.csv",
            "nad_csv": r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\NAD_merged_results.csv",
        }

    def load(self, features: Optional[Dict] = None) -> Dict:
        """載入特徵並建立訓練資料集

        Args:
            features: 如果是None，從檔案載入；否則使用提供的特徵

        Returns:
            包含不同配置的訓練資料集字典
        """
        logger.info("開始載入特徵資料")

        # 初始化資料載入器
        loader = FeatureDataLoader(self.data_path)

        # 載入人口學資料
        demo_processor = self._load_demographics()

        # 準備所有組合的資料集
        datasets = {}

        for embedding_model in self.embedding_models:
            for feature_type in self.feature_types:
                key = f"{embedding_model}_{feature_type}"

                # 載入特徵
                dataset = self._create_dataset(
                    loader, demo_processor, embedding_model, feature_type
                )

                if dataset:
                    datasets[key] = dataset
                    logger.info(f"  {key}: 載入 {len(dataset['X'])} 筆資料")

        return datasets

    def _load_demographics(self) -> Optional[DemographicsProcessor]:
        """載入人口學資料"""
        try:
            processor = DemographicsProcessor()
            processor.load_tables(
                p_source=self.demo_paths["p_csv"],
                acs_source=self.demo_paths["acs_csv"],
                nad_source=self.demo_paths["nad_csv"],
            )
            logger.info("成功載入人口學資料")
            return processor
        except Exception as e:
            logger.warning(f"無法載入人口學資料: {e}")
            return None

    def _create_dataset(
        self,
        loader: FeatureDataLoader,
        demo_processor: Optional[DemographicsProcessor],
        embedding_model: str,
        feature_type: str,
    ) -> Optional[Dict]:
        """建立單一配置的訓練資料集"""

        # 掃描個案
        subjects = loader.scan_subjects(use_all_visits=self.use_all_visits)

        # 應用資料平衡策略
        allowed_ids = None
        demo_lookup = None

        if demo_processor and (self.age_matching or self.cdr_filter):
            balance_config = BalancingConfig(
                enable_age_matching=self.age_matching,
                enable_cdr_filter=self.cdr_filter,
                cdr_threshold=0.5 if self.cdr_filter else None,
            )
            balancer = DataBalancer(demo_processor, balance_config)
            allowed_ids, summary = balancer.balance_groups()
            demo_lookup = demo_processor.build_lookup_table()

        # 載入特徵
        feature_data = loader.load_features(
            subjects, embedding_model=embedding_model, feature_type=feature_type
        )

        # 篩選並準備資料
        X_list, y_list, subject_ids = [], [], []

        for fd in feature_data:
            # 檢查是否在允許清單中
            if allowed_ids:
                from src.utils.id_parser import parse_subject_id

                base_id, _ = parse_subject_id(fd.subject_info.subject_id)
                group = fd.subject_info.group

                if group in allowed_ids:
                    if base_id not in allowed_ids[group]:
                        continue

            features = list(fd.features.values())[0]
            if features is not None:
                X_list.append(features)
                y_list.append(fd.subject_info.label)
                subject_ids.append(fd.subject_info.subject_id)

        if not X_list:
            return None

        import numpy as np

        X = np.array(X_list)
        y = np.array(y_list)

        # 整合人口學特徵
        if demo_lookup:
            integrator = DemographicFeatureIntegrator()
            X = integrator.integrate_features(X, subject_ids, demo_lookup)

        return {
            "X": X,
            "y": y,
            "subject_ids": subject_ids,
            "metadata": {
                "embedding_model": embedding_model,
                "feature_type": feature_type,
                "use_all_visits": self.use_all_visits,
                "age_matching": self.age_matching,
                "cdr_filter": self.cdr_filter,
            },
        }
