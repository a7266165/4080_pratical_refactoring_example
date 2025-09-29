# src/featureloader/selector/demographics.py
"""人口學資料處理"""
from pathlib import Path
from typing import Dict, Optional, Set
import pandas as pd
import numpy as np
import re
import logging
from src.utils.id_parser import parse_subject_id

logger = logging.getLogger(__name__)

class DemographicsLoader:
    """人口學資料處理器"""
    
    def __init__(self):
        self.tables: Dict[str, pd.DataFrame] = {}
        self.lookup_table: Dict[str, Dict[str, float]] = {}
        
    def load_tables(
        self,
        p_source: str,
        acs_source: str,
        nad_source: str
    ) -> Dict[str, pd.DataFrame]:
        """載入年齡表"""
        self.tables = {
            "P": self._read_table(p_source),
            "ACS": self._read_table(acs_source),
            "NAD": self._read_table(nad_source)
        }
        
        # 處理多次訪視（選擇最新）
        for group in self.tables:
            self.tables[group] = self._select_latest_visits(self.tables[group])
        
        logger.info(f"載入人口學資料完成：P={len(self.tables['P'])}人, "
                   f"ACS={len(self.tables['ACS'])}人, NAD={len(self.tables['NAD'])}人")
        
        return self.tables
    
    def _read_table(self, path: str) -> pd.DataFrame:
        """讀取單一表格檔案"""
        p = Path(path)
        
        if p.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        else:
            for encoding in ["utf-8-sig", "utf-8", "cp950", "big5"]:
                try:
                    df = pd.read_csv(p, encoding=encoding)
                    break
                except:
                    continue
            else:
                raise ValueError(f"無法讀取檔案: {path}")
        
        # 清理資料
        df = df.dropna(subset=["ID", "Age"])
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df.dropna(subset=["Age"])
        
        # 處理CDR欄位
        if "Global_CDR" in df.columns:
            df["Global_CDR"] = pd.to_numeric(df["Global_CDR"], errors="coerce")
            logger.info(f"CDR範圍: {df['Global_CDR'].min():.1f} - "
                       f"{df['Global_CDR'].max():.1f}")
        
        # 處理性別欄位
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].apply(self._parse_sex)
        
        return df
    
    def _parse_sex(self, value) -> Optional[float]:
        """解析性別值 (M=1, F=0)"""
        if pd.isna(value):
            return np.nan
        s = str(value).strip().upper()
        if s == "M":
            return 1.0
        elif s == "F":
            return 0.0
        return np.nan
    
    def _select_latest_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        """選擇每個受試者的最新訪視"""
        df = df.copy()
        parsed_ids = df["ID"].apply(lambda x: parse_subject_id(str(x)))
        df["base_id"] = parsed_ids.apply(lambda x: x[0])
        df["visit_number"] = parsed_ids.apply(lambda x: x[1])
        
        latest = df.sort_values("visit_number", ascending=False).groupby("base_id").first()
        latest = latest.reset_index(drop=True)
        
        if "base_id" in latest.columns:
            latest = latest.drop(columns=["base_id", "visit_number"], errors='ignore')
        
        return latest
    
    def build_lookup_table(self) -> Dict[str, Dict[str, float]]:
        """建立人口學資料查詢表"""
        frames = []
        for group in ["P", "ACS", "NAD"]:
            if group in self.tables:
                df = self.tables[group][["ID", "Age"]].copy()
                if "Sex" in self.tables[group].columns:
                    df["Sex"] = self.tables[group]["Sex"]
                frames.append(df)
        
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
        
        # 計算性別眾數
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
        logger.info(f"建立查詢表完成，共 {len(lookup)} 筆資料")
        
        return lookup