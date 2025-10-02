# src/loader.py
"""整合的資料載入與選擇模組"""
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.path_config import DATA_PATHS
from src.utils import parse_subject_id, load_json

logger = logging.getLogger(__name__)


# ========== DataSelector 部分 ==========
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


# ========== DataLoader 部分 ==========
class DataLoader:
    """整合的資料載入器"""
    
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
        self.data_path = Path(DATA_PATHS["features"])
        self.demo_paths = {
            "p_csv": DATA_PATHS["demographics"]["p_csv"],
            "acs_csv": DATA_PATHS["demographics"]["acs_csv"],
            "nad_csv": DATA_PATHS["demographics"]["nad_csv"]
        }
        
        # 初始化資料選擇器
        self.selectors = self._initialize_selectors()
    
    def _initialize_selectors(self) -> Dict[str, DataSelector]:
        """初始化資料選擇器"""
        selectors = {}
        
        if self.cdr_thresholds:
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
        else:
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
        """載入所有配置的資料集"""
        logger.info("開始載入特徵資料...")
        datasets = {}
        
        for selector_key, selector in self.selectors.items():
            logger.info(f"\n處理 {selector_key} 資料集...")
            
            # 掃描個案
            subjects = self._scan_subjects(selector.allowed_ids)
            logger.info(f"  掃描到 {len(subjects)} 個個案")
            
            # 對每個嵌入模型和特徵類型組合
            for embedding_model in self.embedding_models:
                for feature_type in self.feature_types:
                    dataset_key = self._create_dataset_key(
                        embedding_model, feature_type, selector_key
                    )
                    
                    # 載入特徵並準備訓練資料
                    X, y, subject_ids = self._load_and_prepare(
                        subjects,
                        embedding_model,
                        feature_type,
                        selector.lookup_table
                    )
                    
                    if len(X) == 0:
                        logger.warning(f"  {dataset_key}: 無資料")
                        continue
                    
                    datasets[dataset_key] = {
                        "X": X,
                        "y": y,
                        "subject_ids": subject_ids,
                        "metadata": {
                            "embedding_model": embedding_model,
                            "feature_type": feature_type,
                            "use_all_visits": self.use_all_visits,
                            "age_matching": self.age_matching,
                            "cdr_threshold": getattr(selector, 'cdr_threshold', None),
                            "n_samples": len(X),
                            "n_features": X.shape[1],
                            "n_health": np.sum(y == 0),
                            "n_patient": np.sum(y == 1)
                        }
                    }
                    
                    logger.info(f"  {dataset_key}: {len(X)} 樣本, {X.shape[1]} 特徵")
        
        logger.info(f"\n總共載入 {len(datasets)} 個資料集配置")
        return datasets
    
    # ========== 掃描個案（原 FeatureLoader 功能） ==========
    def _scan_subjects(self, allowed_subjects: Dict[str, Set[str]]) -> List[Dict]:
        """掃描並收集個案資料"""
        subjects = []
        
        # 掃描健康組
        health_path = self.data_path / "health"
        for subgroup in ["ACS", "NAD"]:
            subgroup_path = health_path / subgroup
            if subgroup_path.exists():
                subjects.extend(
                    self._scan_group_folders(subgroup_path, subgroup, allowed_subjects.get(subgroup, set()))
                )
        
        # 掃描病患組
        patient_path = self.data_path / "patient"
        if patient_path.exists():
            subjects.extend(
                self._scan_group_folders(patient_path, "P", allowed_subjects.get("P", set()))
            )
        
        return subjects
    
    def _scan_group_folders(self, group_path: Path, group: str, allowed_ids: Set[str]) -> List[Dict]:
        """掃描單一群組的資料夾"""
        subjects = []
        all_folders = [f for f in group_path.iterdir() if f.is_dir()]
        
        # 按受試者ID分組
        subject_visits = {}
        
        for folder in all_folders:
            # 檢查是否在允許清單中
            if allowed_ids and not self._is_allowed(folder.name, allowed_ids):
                continue
            
            # 解析資料夾名稱
            base_id, visit_num = parse_subject_id(folder.name)
            
            # 收集JSON檔案
            json_files = sorted(folder.glob("*_LR_difference.json"))
            if not json_files:
                json_files = sorted(folder.glob("*.json"))
            
            if not json_files:
                continue
            
            # 提取數字ID
            numbers = re.findall(r'\d+', base_id)
            subject_id_num = int(numbers[0]) if numbers else hash(base_id) % 100000
            
            # 加入到訪視字典
            subject_visits.setdefault(subject_id_num, []).append({
                "visit": visit_num,
                "folder": folder,
                "files": json_files,
                "base_id": base_id
            })
        
        # 根據設定選擇訪視
        for subject_id_num, visits in subject_visits.items():
            visits.sort(key=lambda x: x["visit"], reverse=True)
            selected_visits = visits if self.use_all_visits else [visits[0]]
            
            for visit_data in selected_visits:
                subjects.append({
                    "group": group,
                    "subject_id": f"{group}{subject_id_num}",
                    "visit": visit_data["visit"],
                    "feature_paths": visit_data["files"],
                    "label": 1 if group == "P" else 0
                })
        
        return subjects
    
    def _is_allowed(self, folder_name: str, allowed_ids: Set[str]) -> bool:
        """檢查是否在允許清單中"""
        if not allowed_ids:
            return True
        
        base_id, _ = parse_subject_id(folder_name)
        
        return any([
            folder_name in allowed_ids,
            base_id in allowed_ids,
            str(base_id) in allowed_ids,
            str(re.findall(r'\d+', base_id)[0]) in allowed_ids if re.findall(r'\d+', base_id) else False
        ])
    
    # ========== 載入特徵 ==========
    def _load_and_prepare(
        self,
        subjects: List[Dict],
        embedding_model: str,
        feature_type: str,
        lookup_table: Dict
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """載入特徵並準備訓練資料"""
        X_list, y_list, subject_ids = [], [], []
        
        for subject in subjects:
            # 提取特徵
            features = self._extract_features(
                subject["feature_paths"],
                embedding_model,
                feature_type
            )
            
            if features is not None:
                X_list.append(features)
                y_list.append(subject["label"])
                subject_ids.append(subject["subject_id"])
        
        if not X_list:
            return np.array([]), np.array([]), []
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 整合人口學特徵
        if lookup_table:
            X = self._add_demographics(X, subject_ids, lookup_table)
        
        return X, y, subject_ids
    
    def _extract_features(self, json_files: List[Path], embedding_model: str, feature_type: str) -> Optional[np.ndarray]:
        """從JSON檔案中提取並平均特徵向量"""
        vectors = []
        
        for json_file in json_files:
            try:
                data = load_json(json_file)
                
                if feature_type == "difference":
                    feat = data.get("embedding_differences", {}).get(embedding_model)
                elif feature_type == "average":
                    feat = data.get("embedding_averages", {}).get(embedding_model)
                elif feature_type == "relative":
                    feat = data.get("relative_differences", {}).get(embedding_model)
                else:
                    continue
                
                if feat is not None:
                    vectors.append(np.asarray(feat, dtype=float))
            except Exception as e:
                logger.warning(f"讀取 {json_file} 失敗: {e}")
                continue
        
        if not vectors:
            return None
        
        # 檢查維度一致性
        shapes = {v.shape for v in vectors}
        if len(shapes) > 1:
            logger.warning(f"向量維度不一致: {shapes}")
            return None
        
        # 計算平均向量
        return np.vstack(vectors).mean(axis=0)
    
    def _add_demographics(self, X: np.ndarray, subject_ids: List[str], lookup_table: Dict) -> np.ndarray:
        """添加人口學特徵"""
        demo_features = []
        
        for sid in subject_ids:
            meta = lookup_table.get(sid)
            if meta is None:
                base_id, _ = parse_subject_id(sid)
                meta = lookup_table.get(base_id)
            
            age = meta.get("Age", np.nan) if meta else np.nan
            sex = meta.get("Sex", np.nan) if meta else np.nan
            demo_features.append([age, sex])
        
        demo_array = np.array(demo_features)
        
        # 填補缺失值
        age_mean = np.nanmean(demo_array[:, 0])
        if np.isnan(age_mean):
            age_mean = 70
        sex_mode = lookup_table.get("_SEX_MODE_", 0.5)
        
        demo_array[np.isnan(demo_array[:, 0]), 0] = age_mean
        demo_array[np.isnan(demo_array[:, 1]), 1] = sex_mode
        
        # 標準化並結合
        scaler_X = StandardScaler()
        scaler_demo = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        demo_scaled = scaler_demo.fit_transform(demo_array)
        
        return np.hstack([X_scaled, demo_scaled])
    
    def _create_dataset_key(self, embedding_model: str, feature_type: str, selector_key: str) -> str:
        """建立資料集鍵值"""
        if selector_key == "standard":
            return f"{embedding_model}_{feature_type}"
        return f"{embedding_model}_{feature_type}_{selector_key}"