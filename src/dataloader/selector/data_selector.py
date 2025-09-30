# src/dataloader/data_selector.py
"""整合的資料篩選與平衡模組"""
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import pandas as pd
import numpy as np
import re
import logging
from src.utils.utils import parse_subject_id

logger = logging.getLogger(__name__)


class DataSelector:
    """資料篩選器 - 整合人口學載入、CDR篩選、年齡配對功能"""
    
    def __init__(
        self,
        demo_paths: Dict[str, str],
        age_matching: bool = True,
        cdr_filter: bool = False,
        cdr_threshold: Optional[float] = None,
        n_bins: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            demo_paths: 人口學資料檔案路徑
            age_matching: 是否進行年齡配對
            cdr_filter: 是否進行CDR篩選
            cdr_threshold: CDR篩選閾值
            n_bins: 年齡分箱數量
            random_state: 隨機種子
        """
        self.demo_paths = demo_paths
        self.age_matching = age_matching
        self.cdr_filter = cdr_filter
        self.cdr_threshold = cdr_threshold
        self.n_bins = n_bins
        self.random_state = random_state
        
        # 資料儲存
        self.tables: Dict[str, pd.DataFrame] = {}
        self.allowed_ids: Dict[str, Set[str]] = {}
        self.lookup_table: Dict[str, Dict[str, float]] = {}
        self.summary: Optional[pd.DataFrame] = None
        
    def build_selection(self) -> Dict[str, Set[str]]:
        """建立篩選清單（主要方法）
        
        Returns:
            允許的個案ID字典 {"P": {...}, "ACS": {...}, "NAD": {...}}
        """
        logger.info("建立資料篩選清單...")
        
        # Step 1: 載入人口學資料
        self._load_demographics()
        
        # Step 2: 應用CDR篩選
        if self.cdr_filter and self.cdr_threshold is not None:
            self._apply_cdr_filter()
        
        # Step 3: 年齡配對或直接使用所有資料
        if self.age_matching:
            self._apply_age_matching()
        else:
            self._use_all_subjects()
        
        # Step 4: 建立查詢表
        self._build_lookup_table()
        
        logger.info(
            f"篩選完成: P={len(self.allowed_ids.get('P', []))}, "
            f"ACS={len(self.allowed_ids.get('ACS', []))}, "
            f"NAD={len(self.allowed_ids.get('NAD', []))}"
        )
        
        return self.allowed_ids
    
    # ==================== 載入資料 ====================
    def _load_demographics(self):
        """載入三個人口學資料表"""
        self.tables = {
            "P": self._read_table(self.demo_paths["p_csv"]),
            "ACS": self._read_table(self.demo_paths["acs_csv"]),
            "NAD": self._read_table(self.demo_paths["nad_csv"])
        }
        
        # 選擇每個受試者的最新訪視
        for group in self.tables:
            self.tables[group] = self._select_latest_visits(self.tables[group])
        
        logger.info(
            f"載入人口學資料: P={len(self.tables['P'])}人, "
            f"ACS={len(self.tables['ACS'])}人, NAD={len(self.tables['NAD'])}人"
        )
    
    def _read_table(self, path: str) -> pd.DataFrame:
        """讀取單一資料表"""
        p = Path(path)
        
        # 嘗試不同的讀取方式
        if p.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        else:
            # 嘗試多種編碼
            for encoding in ["utf-8-sig", "utf-8", "cp950", "big5"]:
                try:
                    df = pd.read_csv(p, encoding=encoding)
                    break
                except:
                    continue
            else:
                raise ValueError(f"無法讀取檔案: {path}")
        
        # 基本資料清理
        df = df.dropna(subset=["ID", "Age"])
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df.dropna(subset=["Age"])
        
        # 處理CDR欄位（如果存在）
        if "Global_CDR" in df.columns:
            df["Global_CDR"] = pd.to_numeric(df["Global_CDR"], errors="coerce")
            logger.debug(f"CDR範圍: {df['Global_CDR'].min():.1f} - {df['Global_CDR'].max():.1f}")
        
        # 處理性別欄位（如果存在）
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].apply(self._parse_sex)
        
        return df
    
    def _parse_sex(self, value) -> Optional[float]:
        """解析性別值 (M=1, F=0)"""
        if pd.isna(value):
            return np.nan
        s = str(value).strip().upper()
        return 1.0 if s == "M" else 0.0 if s == "F" else np.nan
    
    def _select_latest_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        """選擇每個受試者的最新訪視"""
        df = df.copy()
        
        # 解析ID以提取基礎ID和訪視號
        parsed = df["ID"].apply(lambda x: parse_subject_id(str(x)))
        df["base_id"] = parsed.apply(lambda x: x[0])
        df["visit"] = parsed.apply(lambda x: x[1])
        
        # 保留最新訪視
        latest = df.sort_values("visit", ascending=False).groupby("base_id").first()
        latest = latest.reset_index(drop=True)
        
        # 清理臨時欄位
        if "base_id" in latest.columns:
            latest = latest.drop(columns=["base_id", "visit"], errors="ignore")
        
        return latest
    
    # ==================== CDR篩選 ====================
    def _apply_cdr_filter(self):
        """應用CDR篩選（只篩選P組）"""
        if "Global_CDR" not in self.tables["P"].columns:
            logger.warning("P表中沒有Global_CDR欄位，跳過CDR篩選")
            return
        
        original_count = len(self.tables["P"])
        self.tables["P"] = self.tables["P"][
            self.tables["P"]["Global_CDR"] > self.cdr_threshold
        ].copy()
        
        filtered_count = len(self.tables["P"])
        logger.info(
            f"CDR篩選 (> {self.cdr_threshold}): "
            f"{original_count} -> {filtered_count} "
            f"(保留 {filtered_count/original_count*100:.1f}%)"
        )
    
    # ==================== 年齡配對 ====================
    def _apply_age_matching(self):
        """執行年齡配對"""
        logger.info("執行年齡配對...")
        
        # 準備資料
        all_df = self._prepare_combined_dataframe()
        
        if len(all_df) == 0:
            logger.warning("沒有資料可供配對")
            self.allowed_ids = {"P": set(), "ACS": set(), "NAD": set()}
            return
        
        # 建立年齡分箱
        try:
            all_df["age_bin"] = pd.qcut(
                all_df["Age"], 
                q=self.n_bins, 
                duplicates="drop"
            )
        except ValueError:
            # 如果無法分成指定箱數，自動調整
            n_bins_eff = min(self.n_bins, all_df["Age"].nunique())
            all_df["age_bin"] = pd.qcut(
                all_df["Age"], 
                q=n_bins_eff, 
                duplicates="drop"
            )
        
        # 在每個年齡箱中平衡樣本
        selected_health_ids: Set[str] = set()
        selected_p_ids: Set[str] = set()
        
        rng = np.random.RandomState(self.random_state)
        
        for bin_val in all_df["age_bin"].cat.categories:
            bin_df = all_df[all_df["age_bin"] == bin_val]
            
            health_df = bin_df[bin_df["group"] == "Health"]
            p_df = bin_df[bin_df["group"] == "P"]
            
            n_health = len(health_df)
            n_p = len(p_df)
            
            if n_health == 0 or n_p == 0:
                continue
            
            # 選擇較少的數量作為目標
            target = min(n_health, n_p)
            
            # 隨機抽樣
            if n_health > target:
                health_sample = health_df.sample(n=target, random_state=rng)
            else:
                health_sample = health_df
            
            if n_p > target:
                p_sample = p_df.sample(n=target, random_state=rng)
            else:
                p_sample = p_df
            
            selected_health_ids.update(health_sample["ID"].tolist())
            selected_p_ids.update(p_sample["ID"].tolist())
        
        # 分離ACS和NAD
        acs_ids = set(self.tables["ACS"]["ID"].tolist())
        nad_ids = set(self.tables["NAD"]["ID"].tolist())
        
        self.allowed_ids = {
            "ACS": selected_health_ids & acs_ids,
            "NAD": selected_health_ids & nad_ids,
            "P": selected_p_ids
        }
        
        # 建立統計摘要
        self._create_summary(all_df)
    
    def _use_all_subjects(self):
        """不進行配對，使用所有受試者"""
        self.allowed_ids = {
            "P": set(self.tables["P"]["ID"].tolist()),
            "ACS": set(self.tables["ACS"]["ID"].tolist()),
            "NAD": set(self.tables["NAD"]["ID"].tolist())
        }
        
        all_df = self._prepare_combined_dataframe()
        self._create_summary(all_df)
    
    def _prepare_combined_dataframe(self) -> pd.DataFrame:
        """準備合併的資料框架"""
        frames = []
        
        # 健康組
        for group in ["ACS", "NAD"]:
            if group in self.tables and len(self.tables[group]) > 0:
                df = self.tables[group][["ID", "Age"]].copy()
                df["group"] = "Health"
                df["origin"] = group
                frames.append(df)
        
        # 病患組
        if "P" in self.tables and len(self.tables["P"]) > 0:
            df = self.tables["P"][["ID", "Age"]].copy()
            df["group"] = "P"
            df["origin"] = "P"
            frames.append(df)
        
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()
    
    # ==================== 查詢表與統計 ====================
    def _build_lookup_table(self):
        """建立人口學資料查詢表"""
        frames = []
        for group in ["P", "ACS", "NAD"]:
            if group in self.tables:
                df = self.tables[group][["ID", "Age"]].copy()
                if "Sex" in self.tables[group].columns:
                    df["Sex"] = self.tables[group]["Sex"]
                frames.append(df)
        
        if not frames:
            self.lookup_table = {}
            return
        
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["ID"], keep="first"
        )
        
        lookup = {}
        for _, row in merged.iterrows():
            subject_id = str(row["ID"])
            lookup[subject_id] = {
                "Age": float(row["Age"]),
                "Sex": float(row["Sex"]) if "Sex" in merged.columns and pd.notna(row.get("Sex")) else np.nan
            }
            
            # 同時儲存base_id查詢
            base_id, _ = parse_subject_id(subject_id)
            if base_id != subject_id and base_id not in lookup:
                lookup[base_id] = lookup[subject_id]
        
        # 計算性別眾數（用於填補缺失值）
        if "Sex" in merged.columns:
            sex_values = merged["Sex"].dropna()
            if len(sex_values) > 0:
                mode_value = sex_values.mode()[0] if len(sex_values.mode()) > 0 else np.nan
                lookup["_SEX_MODE_"] = float(mode_value)
            else:
                lookup["_SEX_MODE_"] = np.nan
        else:
            lookup["_SEX_MODE_"] = np.nan
        
        self.lookup_table = lookup
        logger.debug(f"建立查詢表完成，共 {len(lookup)} 筆資料")
    
    def _create_summary(self, all_df: pd.DataFrame):
        """建立統計摘要"""
        stats = []
        
        # 健康組統計
        health_ids = self.allowed_ids.get("ACS", set()) | self.allowed_ids.get("NAD", set())
        if health_ids:
            health_sub = all_df[all_df["ID"].isin(health_ids)]
            if len(health_sub) > 0:
                stats.append({
                    "group": "Health",
                    "n": len(health_ids),
                    "age_mean": health_sub["Age"].mean(),
                    "age_std": health_sub["Age"].std(),
                    "age_min": health_sub["Age"].min(),
                    "age_max": health_sub["Age"].max()
                })
        
        # 病患組統計
        p_ids = self.allowed_ids.get("P", set())
        if p_ids:
            p_sub = all_df[all_df["ID"].isin(p_ids)]
            if len(p_sub) > 0:
                stats.append({
                    "group": "P",
                    "n": len(p_ids),
                    "age_mean": p_sub["Age"].mean(),
                    "age_std": p_sub["Age"].std(),
                    "age_min": p_sub["Age"].min(),
                    "age_max": p_sub["Age"].max()
                })
        
        if stats:
            self.summary = pd.DataFrame(stats)
            logger.info("資料統計摘要:")
            for _, row in self.summary.iterrows():
                logger.info(
                    f"  {row['group']}: n={row['n']:.0f}, "
                    f"age={row['age_mean']:.1f}±{row['age_std']:.1f} "
                    f"({row['age_min']:.0f}-{row['age_max']:.0f})"
                )
        
        logger.info(
            f"  Health組內: ACS={len(self.allowed_ids.get('ACS', []))}, "
            f"NAD={len(self.allowed_ids.get('NAD', []))}"
        )
    
    # ==================== 公開方法 ====================
    def is_allowed(self, subject_id: str, group: str) -> bool:
        """檢查個案是否在允許清單中"""
        if not self.allowed_ids:
            return True
        
        if group not in self.allowed_ids:
            return False
        
        # 處理可能的訪視編號
        base_id, _ = parse_subject_id(subject_id)
        
        return (
            subject_id in self.allowed_ids[group] or 
            base_id in self.allowed_ids[group]
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
                "cdr_threshold": self.cdr_threshold
            }
        }