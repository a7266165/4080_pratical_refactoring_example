# src/dataloader/loader.py
"""資料載入器 - 整合版"""
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from sklearn.preprocessing import StandardScaler

from config.path_config import DATA_PATHS, get_demo_path
from src.dataloader.dataselector import DataSelector
from src.utils.utils import parse_subject_id, load_json

logger = logging.getLogger(__name__)


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