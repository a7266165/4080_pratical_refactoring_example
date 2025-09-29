# src/data/demographics.py
"""人口學資料處理模組"""
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
import logging
from legacy_V2.src.utils.id_parser import parse_subject_id

logger = logging.getLogger(__name__)


@dataclass
class DemographicsConfig:
    """人口學資料配置"""
    use_latest_visit: bool = True  # 是否只使用最新一次訪視


class DemographicsProcessor:
    """人口學資料處理器
    
    負責：
    - 載入和解析年齡表
    - 處理多次訪視資料
    - 建立人口學查詢表
    - CDR篩選
    
    欄位規範：
    - ID: 個案編號
    - Age: 年齡
    - Sex: 性別 (M/F)
    - Global_CDR: CDR評分
    """
    
    def __init__(self, config: Optional[DemographicsConfig] = None):
        self.config = config or DemographicsConfig()
        self.tables: Dict[str, pd.DataFrame] = {}
        self.lookup_table: Dict[str, Dict[str, float]] = {}
        
    def load_tables(
        self,
        p_source: Optional[str] = None,
        acs_source: Optional[str] = None,
        nad_source: Optional[str] = None,
        excel_source: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """載入年齡表
        
        Args:
            p_source: 病患組CSV路徑
            acs_source: ACS組CSV路徑
            nad_source: NAD組CSV路徑
            excel_source: Excel檔案路徑（包含所有組別）
            
        Returns:
            包含各組資料的字典
        """
        if excel_source:
            self.tables = self._load_from_excel(excel_source)
        else:
            if not all([p_source, acs_source, nad_source]):
                raise ValueError("請提供 excel_source 或所有三個CSV路徑")
            
            self.tables = {
                "P": self._read_table(p_source),
                "ACS": self._read_table(acs_source),
                "NAD": self._read_table(nad_source)
            }
        
        # 處理多次訪視（如果需要）
        if self.config.use_latest_visit:
            for group in self.tables:
                self.tables[group] = self._select_latest_visits(self.tables[group])
        
        logger.info(f"載入人口學資料完成：P={len(self.tables['P'])}人, "
                   f"ACS={len(self.tables['ACS'])}人, NAD={len(self.tables['NAD'])}人")
        
        return self.tables
    
    def _read_table(self, path: str) -> pd.DataFrame:
        """讀取單一表格檔案"""
        p = Path(path)
        
        # 嘗試不同的編碼
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
        
        # 驗證必要欄位
        required_columns = ["ID", "Age"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必要欄位: {missing}")
        
        # 清理資料
        df = df.dropna(subset=["ID", "Age"])
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df.dropna(subset=["Age"])
        
        # 處理CDR欄位（如果存在）
        if "Global_CDR" in df.columns:
            df["Global_CDR"] = pd.to_numeric(df["Global_CDR"], errors="coerce")
            logger.info(f"CDR範圍: {df['Global_CDR'].min():.1f} - "
                       f"{df['Global_CDR'].max():.1f}")
        
        # 處理性別欄位（如果存在） - 直接使用M/F格式
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
        else:
            return np.nan
    
    def _select_latest_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        """選擇每個受試者的最新訪視"""
        df = df.copy()
        
        # 解析base_id和visit_number
        parsed_ids = df["ID"].apply(lambda x: parse_subject_id(str(x)))
        df["base_id"] = parsed_ids.apply(lambda x: x[0])
        df["visit_number"] = parsed_ids.apply(lambda x: x[1])
        
        # 按受試者分組，選擇最新訪視
        latest = df.sort_values("visit_number", ascending=False).groupby("base_id").first()
        latest = latest.reset_index(drop=True)
        
        # 清理臨時欄位
        if "base_id" in latest.columns:
            latest = latest.drop(columns=["base_id", "visit_number"])
        
        return latest
    
    def _load_from_excel(self, excel_path: str) -> Dict[str, pd.DataFrame]:
        """從Excel檔案載入所有組別"""
        xl = pd.ExcelFile(excel_path)
        sheets = {name.lower(): name for name in xl.sheet_names}
        
        def find_sheet(keyword: str) -> str:
            for k, v in sheets.items():
                if keyword.lower() in k:
                    return v
            raise ValueError(f"Excel中找不到包含'{keyword}'的工作表")
        
        return {
            "P": self._read_table_from_excel(excel_path, find_sheet("P")),
            "ACS": self._read_table_from_excel(excel_path, find_sheet("ACS")),
            "NAD": self._read_table_from_excel(excel_path, find_sheet("NAD"))
        }
    
    def _read_table_from_excel(self, excel_path: str, sheet_name: str) -> pd.DataFrame:
        """從Excel讀取單一工作表"""
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # 驗證欄位（與 _read_table 相同邏輯）
        required_columns = ["ID", "Age"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"工作表 {sheet_name} 缺少必要欄位: {missing}")
        
        # 清理資料
        df = df.dropna(subset=["ID", "Age"])
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df.dropna(subset=["Age"])
        
        if "Global_CDR" in df.columns:
            df["Global_CDR"] = pd.to_numeric(df["Global_CDR"], errors="coerce")
        
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].apply(self._parse_sex)
        
        return df
    
    def filter_by_cdr(
        self,
        cdr_threshold: float,
        group: str = "P"
    ) -> pd.DataFrame:
        """根據CDR值篩選資料
        
        Args:
            cdr_threshold: CDR閾值
            group: 要篩選的組別
            
        Returns:
            篩選後的資料框
        """
        if group not in self.tables:
            raise ValueError(f"組別 {group} 不存在")
        
        df = self.tables[group]
        
        if "Global_CDR" not in df.columns:
            logger.warning(f"{group} 組沒有Global_CDR欄位，返回原始資料")
            return df
        
        original_count = len(df)
        filtered_df = df[df["Global_CDR"] > cdr_threshold].copy()
        filtered_count = len(filtered_df)
        
        logger.info(f"CDR篩選 (>{cdr_threshold}): {original_count} -> {filtered_count} "
                   f"({filtered_count/original_count*100:.1f}%)")
        
        return filtered_df
    
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
        
        # 建立查詢表
        lookup = {}
        for _, row in merged.iterrows():
            subject_id = str(row["ID"])
            
            # 處理性別值（可能是 M/F 字串或已經是 0/1）
            sex_value = np.nan
            if "Sex" in merged.columns and pd.notna(row.get("Sex")):
                sex_raw = row["Sex"]
                if isinstance(sex_raw, str):
                    # 如果是字串，使用 _parse_sex 處理
                    sex_value = self._parse_sex(sex_raw)
                else:
                    # 如果已經是數值，直接使用
                    sex_value = float(sex_raw)
            
            lookup[subject_id] = {
                "Age": float(row["Age"]),
                "Sex": sex_value
            }
            
            # 同時儲存base_id查詢（使用統一工具）
            base_id, _ = parse_subject_id(subject_id)
            if base_id != subject_id and base_id not in lookup:
                lookup[base_id] = lookup[subject_id]
        
        # 計算性別眾數（用於填補缺失值）
        if "Sex" in merged.columns:
            # 先轉換所有性別值
            sex_values = []
            for val in merged["Sex"].dropna():
                if isinstance(val, str):
                    parsed = self._parse_sex(val)
                    if not np.isnan(parsed):
                        sex_values.append(parsed)
                else:
                    sex_values.append(float(val))
            
            if sex_values:
                # 計算眾數
                from collections import Counter
                counter = Counter(sex_values)
                mode_value = counter.most_common(1)[0][0]
                lookup["_SEX_MODE_"] = float(mode_value)
            else:
                lookup["_SEX_MODE_"] = np.nan
        else:
            lookup["_SEX_MODE_"] = np.nan
        
        self.lookup_table = lookup
        logger.info(f"建立查詢表完成，共 {len(lookup)} 筆資料")
        
        return lookup