# src/featureloader/selector/age_balancing.py
"""年齡平衡策略"""
from typing import Dict, Set, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataBalancer:
    """資料平衡器"""
    
    def __init__(
        self,
        demographics_processor,
        enable_age_matching: bool = True,
        enable_cdr_filter: bool = False,
        cdr_threshold: float = 0.5,
        n_bins: int = 5,
        random_state: int = 42
    ):
        self.demo_processor = demographics_processor
        self.enable_age_matching = enable_age_matching
        self.enable_cdr_filter = enable_cdr_filter
        self.cdr_threshold = cdr_threshold
        self.n_bins = n_bins
        self.rng = np.random.RandomState(random_state)
        
    def balance_groups(self) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
        """執行組別平衡"""
        tables = self.demo_processor.tables
        
        # CDR篩選
        if self.enable_cdr_filter and self.cdr_threshold is not None:
            p_df = tables["P"]
            if "Global_CDR" in p_df.columns:
                tables["P"] = p_df[p_df["Global_CDR"] > self.cdr_threshold].copy()
                
                if len(tables["P"]) == 0:
                    logger.warning(f"CDR篩選後P組無資料")
                    return {"ACS": set(), "NAD": set(), "P": set()}, pd.DataFrame()
        
        # 年齡配對
        if self.enable_age_matching:
            return self._age_balance(tables)
        else:
            return self._no_balance(tables)
    
    def _age_balance(self, tables: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
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
        try:
            all_df["age_bin"] = pd.qcut(
                all_df["Age"],
                q=self.n_bins,
                duplicates="drop"
            )
        except ValueError:
            n_bins_eff = min(self.n_bins, all_df["Age"].nunique())
            all_df["age_bin"] = pd.qcut(
                all_df["Age"],
                q=n_bins_eff,
                duplicates="drop"
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
            
            target = min(n_health, n_p)
            
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
    
    def _no_balance(self, tables: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
        """不進行平衡"""
        allowed_ids = {
            "ACS": set(tables["ACS"]["ID"].tolist()),
            "NAD": set(tables["NAD"]["ID"].tolist()),
            "P": set(tables["P"]["ID"].tolist())
        }
        
        all_dfs = []
        for group, df in tables.items():
            temp_df = df[["ID", "Age"]].copy()
            temp_df["group"] = group
            all_dfs.append(temp_df)
        
        all_df = pd.concat(all_dfs, ignore_index=True)
        summary = self._create_summary(all_df, allowed_ids)
        
        return allowed_ids, summary
    
    def _create_summary(self, all_df: pd.DataFrame, allowed_ids: Dict[str, Set[str]]) -> pd.DataFrame:
        """建立統計摘要"""
        stats = []
        
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
        
        logger.info(f"\n資料平衡統計:")
        for _, row in summary_df.iterrows():
            logger.info(f"  {row['group']}: n={row['n']:.0f}, "
                       f"age={row['age_mean']:.1f}±{row['age_std']:.1f} "
                       f"({row['age_min']:.0f}-{row['age_max']:.0f})")
        
        logger.info(f"  Health組內: ACS={len(allowed_ids['ACS'])}, "
                   f"NAD={len(allowed_ids['NAD'])}")
        
        return summary_df