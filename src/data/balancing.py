# src/data/balancing.py
"""資料平衡策略模組"""
from typing import Dict, Set, Tuple, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.data.demographics import DemographicsProcessor
import logging

logger = logging.getLogger(__name__)


@dataclass
class BalancingConfig:
    """資料平衡配置"""
    enable_age_matching: bool = True
    enable_cdr_filter: bool = False
    cdr_threshold: float = 0.5
    n_bins: int = 5
    random_state: int = 42
    method: str = "quantile"  # "quantile" or "equal_width"


class DataBalancer:
    """資料平衡器
    
    負責：
    - 年齡配對
    - CDR篩選
    - 組別平衡
    """
    
    def __init__(
        self,
        demographics_processor: DemographicsProcessor,
        config: Optional[BalancingConfig] = None
    ):
        self.demo_processor = demographics_processor
        self.config = config or BalancingConfig()
        self.rng = np.random.RandomState(self.config.random_state)
        
    def balance_groups(
        self,
        groups: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
        """執行組別平衡
        
        Args:
            groups: 要平衡的組別，預設為 ["ACS", "NAD"] vs ["P"]
            
        Returns:
            - allowed_ids: 各組允許使用的ID集合
            - summary: 統計摘要
        """
        if groups is None:
            groups = ["ACS", "NAD", "P"]
        
        tables = self.demo_processor.tables
        
        # CDR篩選
        if self.config.enable_cdr_filter and self.config.cdr_threshold is not None:
            tables["P"] = self.demo_processor.filter_by_cdr(
                self.config.cdr_threshold, "P"
            )
            
            if len(tables["P"]) == 0:
                logger.warning(f"CDR篩選後P組無資料")
                return {"ACS": set(), "NAD": set(), "P": set()}, pd.DataFrame()
        
        # 年齡配對
        if self.config.enable_age_matching:
            return self._age_balance(tables)
        else:
            return self._no_balance(tables)
    
    def _age_balance(
        self,
        tables: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
        """執行年齡配對"""
        logger.info("執行年齡配對...")
        
        # 合併健康組
        acs_df = tables["ACS"][["ID", "Age"]].copy()
        acs_df["origin"] = "ACS"
        nad_df = tables["NAD"][["ID", "Age"]].copy()
        nad_df["origin"] = "NAD"
        health_df = pd.concat([acs_df, nad_df], ignore_index=True)
        health_df["group"] = "Health"
        
        # 病患組
        p_df = tables["P"][["ID", "Age"]].copy()
        p_df["group"] = "P"
        p_df["origin"] = "P"
        
        # 合併所有資料
        all_df = pd.concat([health_df, p_df], ignore_index=True)
        
        # 建立年齡分箱
        if self.config.method == "quantile":
            try:
                all_df["age_bin"] = pd.qcut(
                    all_df["Age"],
                    q=self.config.n_bins,
                    duplicates="drop"
                )
            except ValueError:
                # 如果分箱失敗，減少箱數
                n_bins_eff = min(self.config.n_bins, all_df["Age"].nunique())
                all_df["age_bin"] = pd.qcut(
                    all_df["Age"],
                    q=n_bins_eff,
                    duplicates="drop"
                )
        else:  # equal_width
            all_df["age_bin"] = pd.cut(
                all_df["Age"],
                bins=self.config.n_bins
            )
        
        # 在每個箱中平衡樣本
        selected_health_ids: Set[str] = set()
        selected_p_ids: Set[str] = set()
        
        for bin_val in all_df["age_bin"].cat.categories:
            bin_df = all_df[all_df["age_bin"] == bin_val]
            
            n_health = bin_df[bin_df["group"] == "Health"].shape[0]
            n_p = bin_df[bin_df["group"] == "P"].shape[0]
            
            if n_health == 0 or n_p == 0:
                continue
            
            # 選擇較少的數量
            target = min(n_health, n_p)
            
            # 隨機抽樣
            health_pool = bin_df[bin_df["group"] == "Health"]
            p_pool = bin_df[bin_df["group"] == "P"]
            
            if len(health_pool) > target:
                pick_h = health_pool.sample(n=target, random_state=self.rng)
            else:
                pick_h = health_pool
                
            if len(p_pool) > target:
                pick_p = p_pool.sample(n=target, random_state=self.rng)
            else:
                pick_p = p_pool
            
            selected_health_ids.update(pick_h["ID"].tolist())
            selected_p_ids.update(pick_p["ID"].tolist())
        
        # 分離ACS和NAD
        acs_ids_all = set(tables["ACS"]["ID"].tolist())
        nad_ids_all = set(tables["NAD"]["ID"].tolist())
        
        selected_acs = selected_health_ids & acs_ids_all
        selected_nad = selected_health_ids & nad_ids_all
        
        allowed_ids = {
            "ACS": selected_acs,
            "NAD": selected_nad,
            "P": selected_p_ids
        }
        
        # 建立統計摘要
        summary = self._create_summary(all_df, allowed_ids)
        
        return allowed_ids, summary
    
    def _no_balance(
        self,
        tables: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
        """不進行平衡，返回所有ID"""
        logger.info("跳過年齡配對，使用所有資料")
        
        allowed_ids = {
            "ACS": set(tables["ACS"]["ID"].tolist()),
            "NAD": set(tables["NAD"]["ID"].tolist()),
            "P": set(tables["P"]["ID"].tolist())
        }
        
        # 建立統計摘要
        all_dfs = []
        for group, df in tables.items():
            temp_df = df[["ID", "Age"]].copy()
            temp_df["group"] = group
            all_dfs.append(temp_df)
        
        all_df = pd.concat(all_dfs, ignore_index=True)
        summary = self._create_summary(all_df, allowed_ids)
        
        return allowed_ids, summary
    
    def _create_summary(
        self,
        all_df: pd.DataFrame,
        allowed_ids: Dict[str, Set[str]]
    ) -> pd.DataFrame:
        """建立統計摘要"""
        stats = []
        
        # 健康組統計
        health_ids = allowed_ids["ACS"] | allowed_ids["NAD"]
        if health_ids:
            health_sub = all_df[all_df["ID"].isin(health_ids)]
            stats.append({
                "group": "Health",
                "n": len(health_ids),
                "age_mean": health_sub["Age"].mean(),
                "age_std": health_sub["Age"].std(),
                "age_min": health_sub["Age"].min(),
                "age_max": health_sub["Age"].max()
            })
        
        # 病患組統計
        if allowed_ids["P"]:
            p_sub = all_df[all_df["ID"].isin(allowed_ids["P"])]
            stats.append({
                "group": "P",
                "n": len(allowed_ids["P"]),
                "age_mean": p_sub["Age"].mean(),
                "age_std": p_sub["Age"].std(),
                "age_min": p_sub["Age"].min(),
                "age_max": p_sub["Age"].max()
            })
        
        summary_df = pd.DataFrame(stats)
        
        # 輸出統計資訊
        cdr_str = f" (CDR>{self.config.cdr_threshold})" if self.config.enable_cdr_filter else ""
        logger.info(f"\n資料平衡統計{cdr_str}:")
        for _, row in summary_df.iterrows():
            logger.info(f"  {row['group']}: n={row['n']}, "
                       f"age={row['age_mean']:.1f}±{row['age_std']:.1f} "
                       f"({row['age_min']:.0f}-{row['age_max']:.0f})")
        
        if "Health" in summary_df["group"].values:
            logger.info(f"  Health組內: ACS={len(allowed_ids['ACS'])}, "
                       f"NAD={len(allowed_ids['NAD'])}")
        
        return summary_df