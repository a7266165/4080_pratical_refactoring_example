# src/data/loader.py
"""資料載入器"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from .structures import SubjectInfo, SubjectFeature, DatasetInfo
from src.utils.id_parser import parse_subject_id

class FeatureDataLoader:
    """特徵資料載入器"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self._validate_path()
        
    def _validate_path(self):
        """驗證資料路徑與基本結構"""
        if not self.base_path.exists():
            raise ValueError(f"資料路徑不存在: {self.base_path}")
            
        health_path = self.base_path / "health"
        patient_path = self.base_path / "patient"
        if not health_path.exists() or not patient_path.exists():
            raise ValueError("資料結構不完整，缺少 health/ 或 patient/ 資料夾")
    
    def scan_subjects(self, use_all_visits: bool = False) -> List[SubjectInfo]:
        """掃描所有個案
        
        Args:
            use_all_visits: True=使用所有次數, False=只用最新一次
        """
        subjects: List[SubjectInfo] = []
        
        # 掃描健康組
        health_path = self.base_path / "health"
        for subgroup in ["ACS", "NAD"]:
            subgroup_path = health_path / subgroup
            if subgroup_path.exists():
                subjects.extend(self._scan_subgroup(
                    subgroup_path, 
                    group=subgroup,
                    use_all_visits=use_all_visits
                ))
        
        # 掃描病患組
        patient_path = self.base_path / "patient"
        if patient_path.exists():
            subjects.extend(self._scan_subgroup(
                patient_path,
                group="P",
                use_all_visits=use_all_visits
            ))
        
        return subjects
    
    def _scan_subgroup(self, path: Path, group: str, use_all_visits: bool) -> List[SubjectInfo]:
        """掃描子群組資料夾，建立 SubjectInfo 清單"""
        subjects: List[SubjectInfo] = []
        subject_folders = [f for f in path.iterdir() if f.is_dir()]
        
        # 先依 subject（個案）彙整各次訪視的檔案
        subject_dict: Dict[int, List[Dict]] = {}
        for folder in subject_folders:
            subj_id_num, visit_num = self._parse_folder_name(folder.name, fallback_group=group)
            json_files = self._collect_subject_jsons(folder)
            if not json_files:
                continue
            
            subject_dict.setdefault(subj_id_num, []).append({
                "visit": visit_num,
                "folder": folder,
                "files": json_files,
            })
        
        # 依規則挑選訪視
        for subj_id_num, visits in subject_dict.items():
            visits.sort(key=lambda x: x["visit"], reverse=True)
            selected = visits if use_all_visits else [visits[0]]
            for v in selected:
                subjects.append(SubjectInfo(
                    group=group,
                    id=subj_id_num,
                    visit=v["visit"],
                    feature_paths=v["files"],
                ))
        
        return subjects
    
    def _parse_folder_name(self, folder_name: str, fallback_group: str) -> Tuple[int, int]:
        """解析資料夾名稱取得 (subject_id_number, visit_number)
        
        使用統一的ID解析工具
        """
        base_id, visit_number = parse_subject_id(folder_name)
        
        # 提取數字部分作為subject_id_number
        # 例如 "ACS12" -> 12, "P1" -> 1
        numbers = re.findall(r'\d+', base_id)
        if numbers:
            subject_id_number = int(numbers[0])
        else:
            subject_id_number = 0
            
        return subject_id_number, visit_number
    
    def _collect_subject_jsons(self, folder: Path) -> List[Path]:
        """收集該訪視資料夾中的 JSON 特徵檔
        
        優先匹配常見命名（*_LR_*.json），若沒有則回退為所有 .json。
        """
        # 常見：*_LR_difference.json / *_LR_average.json / *_LR_relative.json
        preferred = sorted(folder.glob("*_LR_*.json"))
        if preferred:
            return preferred
        return sorted(folder.glob("*.json"))
    
    def load_features(
        self, 
        subjects: List[SubjectInfo], 
        embedding_model: str,
        feature_type: str = "difference"
    ) -> List[SubjectFeature]:
        """載入特徵資料
        
        Args:
            subjects: 個案清單
            embedding_model: 'vggface', 'arcface', etc.
            feature_type: 'difference', 'average', 'relative'
        """
        feature_data_list: List[SubjectFeature] = []
        
        for subject in subjects:
            features = self._extract_subject_features(
                subject.feature_paths, 
                embedding_model, 
                feature_type
            )
            if features is not None:
                feature_data_list.append(SubjectFeature(
                    subject_info=subject,
                    features={embedding_model: features},
                    feature_type=feature_type
                ))
        
        return feature_data_list
    
    def _extract_subject_features(
        self, 
        json_files: List[Path],
        embedding_model: str,
        feature_type: str
    ) -> Optional[np.ndarray]:
        """從個案的 JSON 檔案中提取特徵，並做平均聚合"""
        vectors: List[np.ndarray] = []
        
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 根據特徵類型提取
            if feature_type == "difference":
                feat = data.get("embedding_differences", {}).get(embedding_model)
            elif feature_type == "average":
                feat = data.get("embedding_averages", {}).get(embedding_model)
            elif feature_type == "relative":
                feat = data.get("relative_differences", {}).get(embedding_model)
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
            
            if feat is not None:
                vec = np.asarray(feat, dtype=float)
                vectors.append(vec)
        
        if not vectors:
            return None
        
        # 維度一致性檢查
        shapes = {v.shape for v in vectors}
        if len(shapes) > 1:
            print(f"警告：向量維度不一致: {shapes}")
            return None
        
        # 平均匯總
        stacked = np.vstack(vectors)
        mean_vec = stacked.mean(axis=0)
        return mean_vec
    
    def get_dataset_info(self, feature_data_list: List[SubjectFeature]) -> DatasetInfo:
        """取得資料集統計資訊"""
        groups: Dict[str, int] = {}
        subjects = set()
        
        for fd in feature_data_list:
            si = fd.subject_info
            groups[si.group] = groups.get(si.group, 0) + 1
            subjects.add(si.subject_id)
        
        n_health = sum(1 for fd in feature_data_list if fd.subject_info.label == 0)
        n_patient = sum(1 for fd in feature_data_list if fd.subject_info.label == 1)
        
        # 特徵維度
        feature_dim = 0
        if feature_data_list:
            first_features = list(feature_data_list[0].features.values())[0]
            feature_dim = int(first_features.shape[0]) if hasattr(first_features, "shape") else len(first_features)
        
        return DatasetInfo(
            n_samples=len(feature_data_list),
            n_subjects=len(subjects),
            n_health=n_health,
            n_patient=n_patient,
            groups=groups,
            feature_dim=feature_dim
        )
