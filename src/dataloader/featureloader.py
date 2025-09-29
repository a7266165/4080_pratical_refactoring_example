# src/dataloader/featureloader.py
"""特徵資料載入器（支援預篩選）"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from src.data_structure import SubjectInfo, SubjectFeature
from src.utils.utils import parse_subject_id, load_json
import logging

logger = logging.getLogger(__name__)


class FeatureLoader:
    """特徵資料載入器（支援預篩選）"""

    def __init__(
        self,
        base_path: str,
        allowed_subjects: Optional[Dict[str, Set[str]]] = None,
        use_all_visits: bool = False,
    ):
        self.base_path = Path(base_path)
        self.allowed_subjects = allowed_subjects or {}
        self.use_all_visits = use_all_visits
        self._validate_path()

    def _validate_path(self):
        """驗證資料路徑與基本結構"""
        if not self.base_path.exists():
            raise ValueError(f"資料路徑不存在: {self.base_path}")

        health_path = self.base_path / "health"
        patient_path = self.base_path / "patient"

        if not health_path.exists() or not patient_path.exists():
            raise ValueError("資料結構不完整，缺少 health/ 或 patient/ 資料夾")

    def scan_allowed_subjects(self) -> List[SubjectInfo]:
        """只掃描允許清單中的個案"""
        subjects: List[SubjectInfo] = []

        # 掃描健康組
        health_path = self.base_path / "health"
        for subgroup in ["ACS", "NAD"]:
            if subgroup in self.allowed_subjects:
                subgroup_path = health_path / subgroup
                if subgroup_path.exists():
                    subjects.extend(
                        self._scan_filtered_subgroup(
                            subgroup_path,
                            group=subgroup,
                            allowed_ids=self.allowed_subjects[subgroup],
                        )
                    )

        # 掃描病患組
        if "P" in self.allowed_subjects:
            patient_path = self.base_path / "patient"
            if patient_path.exists():
                subjects.extend(
                    self._scan_filtered_subgroup(
                        patient_path, group="P", allowed_ids=self.allowed_subjects["P"]
                    )
                )

        logger.info(f"掃描到 {len(subjects)} 個符合條件的個案")
        return subjects

    def scan_subjects(self, use_all_visits: bool = False) -> List[SubjectInfo]:
        """掃描所有個案（向後相容用，建議使用 scan_allowed_subjects）"""
        if self.allowed_subjects:
            # 如果有允許清單，使用篩選版本
            return self.scan_allowed_subjects()

        # 原始版本：掃描所有個案
        subjects: List[SubjectInfo] = []

        # 掃描健康組
        health_path = self.base_path / "health"
        for subgroup in ["ACS", "NAD"]:
            subgroup_path = health_path / subgroup
            if subgroup_path.exists():
                subjects.extend(
                    self._scan_subgroup(
                        subgroup_path, group=subgroup, use_all_visits=use_all_visits
                    )
                )

        # 掃描病患組
        patient_path = self.base_path / "patient"
        if patient_path.exists():
            subjects.extend(
                self._scan_subgroup(
                    patient_path, group="P", use_all_visits=use_all_visits
                )
            )

        return subjects

    def _scan_filtered_subgroup(
        self, path: Path, group: str, allowed_ids: Set[str]
    ) -> List[SubjectInfo]:
        """只掃描允許清單中的子群組資料夾"""
        subjects: List[SubjectInfo] = []
        subject_dict: Dict[int, List[Dict]] = {}

        for folder in path.iterdir():
            if not folder.is_dir():
                continue

            # 解析資料夾名稱
            subj_id_num, visit_num = self._parse_folder_name(
                folder.name, fallback_group=group
            )

            # 檢查是否在允許清單中
            base_id, _ = parse_subject_id(folder.name)
            if not self._is_allowed(
                folder.name, base_id, str(subj_id_num), allowed_ids
            ):
                continue

            # 這個個案在允許清單中，載入它
            json_files = self._collect_subject_jsons(folder)
            if json_files:
                subject_dict.setdefault(subj_id_num, []).append(
                    {
                        "visit": visit_num,
                        "folder": folder,
                        "files": json_files,
                    }
                )

        # 依規則挑選訪視
        for subj_id_num, visits in subject_dict.items():
            visits.sort(key=lambda x: x["visit"], reverse=True)
            selected = visits if self.use_all_visits else [visits[0]]

            for v in selected:
                subjects.append(
                    SubjectInfo(
                        group=group,
                        id=subj_id_num,
                        visit=v["visit"],
                        feature_paths=v["files"],
                    )
                )

        return subjects

    def _is_allowed(
        self, folder_id: str, base_id: str, subj_id_num: str, allowed_ids: Set[str]
    ) -> bool:
        """檢查ID是否在允許清單中（支援多種ID格式）"""
        # 檢查各種可能的ID格式
        return any(
            [
                folder_id in allowed_ids,
                base_id in allowed_ids,
                subj_id_num in allowed_ids,
                str(base_id) in allowed_ids,
                str(subj_id_num) in allowed_ids,
            ]
        )

    def _scan_subgroup(
        self, path: Path, group: str, use_all_visits: bool
    ) -> List[SubjectInfo]:
        """掃描子群組資料夾（原始版本）"""
        subjects: List[SubjectInfo] = []
        subject_folders = [f for f in path.iterdir() if f.is_dir()]

        # 先依 subject 彙整各次訪視的檔案
        subject_dict: Dict[int, List[Dict]] = {}

        for folder in subject_folders:
            subj_id_num, visit_num = self._parse_folder_name(
                folder.name, fallback_group=group
            )
            json_files = self._collect_subject_jsons(folder)

            if not json_files:
                continue

            subject_dict.setdefault(subj_id_num, []).append(
                {
                    "visit": visit_num,
                    "folder": folder,
                    "files": json_files,
                }
            )

        # 依規則挑選訪視
        for subj_id_num, visits in subject_dict.items():
            visits.sort(key=lambda x: x["visit"], reverse=True)
            selected = visits if use_all_visits else [visits[0]]

            for v in selected:
                subjects.append(
                    SubjectInfo(
                        group=group,
                        id=subj_id_num,
                        visit=v["visit"],
                        feature_paths=v["files"],
                    )
                )

        return subjects

    def _parse_folder_name(
        self, folder_name: str, fallback_group: str
    ) -> Tuple[int, int]:
        """解析資料夾名稱"""
        base_id, visit_number = parse_subject_id(folder_name)

        # 提取數字部分作為subject_id_number
        numbers = re.findall(r"\d+", base_id)
        if numbers:
            subject_id_number = int(numbers[0])
        else:
            subject_id_number = 0

        return subject_id_number, visit_number

    def _collect_subject_jsons(self, folder: Path) -> List[Path]:
        """收集該訪視資料夾中的 JSON 特徵檔"""
        preferred = sorted(folder.glob("*_LR_*.json"))
        if preferred:
            return preferred
        return sorted(folder.glob("*.json"))

    def load_features(
        self,
        subjects: List[SubjectInfo],
        embedding_model: str,
        feature_type: str = "difference",
    ) -> List[SubjectFeature]:
        """載入特徵資料"""
        feature_data_list: List[SubjectFeature] = []

        for subject in subjects:
            features = self._extract_subject_features(
                subject.feature_paths, embedding_model, feature_type
            )

            if features is not None:
                feature_data_list.append(
                    SubjectFeature(
                        subject_info=subject,
                        features={embedding_model: features},
                        feature_type=feature_type,
                    )
                )

        return feature_data_list

    def _extract_subject_features(
        self, json_files: List[Path], embedding_model: str, feature_type: str
    ) -> Optional[np.ndarray]:
        """從個案的 JSON 檔案中提取特徵"""
        vectors: List[np.ndarray] = []

        for json_file in json_files:
            data = load_json(json_file)

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
            logger.warning(f"向量維度不一致: {shapes}")
            return None

        # 平均匯總
        stacked = np.vstack(vectors)
        mean_vec = stacked.mean(axis=0)

        return mean_vec
