# src/dataloader/selector/subject_filter.py
"""統一的個案篩選器"""
from typing import Dict, Set, Optional
import pandas as pd
import logging
from src.dataloader.selector.demographics import DemographicsLoader
from src.dataloader.selector.age_balancing import DataBalancer
from src.utils.utils import parse_subject_id

logger = logging.getLogger(__name__)


class SubjectFilter:
    """個案篩選器

    在載入資料前就決定哪些個案要被載入
    """

    def __init__(
        self,
        demo_paths: Dict,
        age_matching: bool = True,
        cdr_filter: bool = False,
        cdr_threshold: Optional[float] = None,
        use_all_visits: bool = False,
        random_state: int = 42,
    ):
        """
        初始化個案篩選器

        Args:
            demo_paths: 人口學資料檔案路徑
            age_matching: 是否進行年齡配對
            cdr_filter: 是否進行CDR篩選
            cdr_threshold: CDR篩選閾值（只有在cdr_filter=True時使用）
            use_all_visits: 是否使用所有訪視
            random_state: 隨機種子
        """
        self.demo_paths = demo_paths
        self.age_matching = age_matching
        self.cdr_filter = cdr_filter
        self.cdr_threshold = cdr_threshold
        self.use_all_visits = use_all_visits
        self.random_state = random_state

        self.allowed_ids: Dict[str, Set[str]] = {}
        self.demographics_lookup: Optional[Dict] = None
        self.summary: Optional[pd.DataFrame] = None

    def build_allowed_list(self):
        """建立允許的個案清單"""
        logger.info("建立個案篩選清單...")

        # 載入人口學資料
        demographics_loader = DemographicsLoader()
        demographics_loader.load_tables(
            p_source=self.demo_paths["p_csv"],
            acs_source=self.demo_paths["acs_csv"],
            nad_source=self.demo_paths["nad_csv"],
        )

        # 如果需要篩選，使用DataBalancer
        if self.age_matching or self.cdr_filter:
            balancer = DataBalancer(
                demographics_loader,
                enable_age_matching=self.age_matching,
                enable_cdr_filter=self.cdr_filter,
                cdr_threshold=self.cdr_threshold,
                random_state=self.random_state,
            )

            self.allowed_ids, self.summary = balancer.balance_groups()

        else:
            # 不篩選，允許所有個案
            self.allowed_ids = {
                "P": set(demographics_loader.tables["P"]["ID"].tolist()),
                "ACS": set(demographics_loader.tables["ACS"]["ID"].tolist()),
                "NAD": set(demographics_loader.tables["NAD"]["ID"].tolist()),
            }

        # 建立查詢表供後續使用
        self.demographics_lookup = demographics_loader.build_lookup_table()

        logger.info(
            f"篩選清單建立完成: "
            f"總共 {sum(len(ids) for ids in self.allowed_ids.values())} 個個案"
        )

    def is_allowed(self, subject_id: str, group: str) -> bool:
        """檢查個案是否在允許清單中"""
        if not self.allowed_ids:
            return True  # 如果沒有篩選，允許所有

        if group not in self.allowed_ids:
            return False

        # 處理可能的訪視編號
        base_id, _ = parse_subject_id(subject_id)

        return (
            subject_id in self.allowed_ids[group] or base_id in self.allowed_ids[group]
        )

    def get_summary(self) -> Dict:
        """取得篩選統計摘要"""
        return {
            "total_allowed": sum(len(ids) for ids in self.allowed_ids.values()),
            "P_count": len(self.allowed_ids.get("P", [])),
            "ACS_count": len(self.allowed_ids.get("ACS", [])),
            "NAD_count": len(self.allowed_ids.get("NAD", [])),
            "filters_applied": {
                "age_matching": self.age_matching,
                "cdr_filter": self.cdr_filter,
                "cdr_threshold": self.cdr_threshold,
            },
        }
