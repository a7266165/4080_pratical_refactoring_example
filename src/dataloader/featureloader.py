# src/dataloader/featureloader.py
"""簡化版特徵資料載入器"""
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
    """特徵資料載入器"""
    
    def __init__(
        self,
        base_path: str,
        allowed_subjects: Optional[Dict[str, Set[str]]] = None,
        use_all_visits: bool = False
    ):
        """
        Args:
            base_path: 資料根目錄
            allowed_subjects: 允許的個案清單 {"P": {...}, "ACS": {...}, "NAD": {...}}
            use_all_visits: 是否載入所有訪視（False=只用最新）
        """
        self.base_path = Path(base_path)
        self.allowed_subjects = allowed_subjects or {}
        self.use_all_visits = use_all_visits
        self._validate_path()
    
    def _validate_path(self):
        """驗證資料路徑結構"""
        if not self.base_path.exists():
            raise ValueError(f"資料路徑不存在: {self.base_path}")
        
        health_path = self.base_path / "health"
        patient_path = self.base_path / "patient"
        
        if not health_path.exists() or not patient_path.exists():
            raise ValueError("資料結構不完整，缺少 health/ 或 patient/ 資料夾")
    
    # ==================== 掃描個案 ====================
    def scan_subjects(self) -> List[SubjectInfo]:
        """統一的掃描方法"""
        subjects: List[SubjectInfo] = []
        
        # 掃描健康組 (ACS, NAD)
        health_path = self.base_path / "health"
        for subgroup in ["ACS", "NAD"]:
            subgroup_path = health_path / subgroup
            if subgroup_path.exists():
                subjects.extend(
                    self._scan_group_folders(subgroup_path, group=subgroup)
                )
        
        # 掃描病患組 (P)
        patient_path = self.base_path / "patient"
        if patient_path.exists():
            subjects.extend(
                self._scan_group_folders(patient_path, group="P")
            )
        
        logger.info(f"掃描到 {len(subjects)} 個個案")
        return subjects
    
    def _scan_group_folders(
        self,
        group_path: Path,
        group: str
    ) -> List[SubjectInfo]:
        """掃描單一群組的資料夾"""
        subjects: List[SubjectInfo] = []
        
        # 收集該群組下的所有資料夾
        all_folders = [f for f in group_path.iterdir() if f.is_dir()]
        
        # 按受試者ID分組
        subject_visits: Dict[int, List[Dict]] = {}
        
        for folder in all_folders:
            # 檢查是否在允許清單中
            if not self._is_allowed(folder.name, group):
                continue
            
            # 解析資料夾名稱
            subject_id, visit_num = self._parse_folder_name(folder.name, group)
            
            # 收集JSON檔案
            json_files = self._collect_json_files(folder)
            if not json_files:
                continue
            
            # 加入到受試者訪視字典
            subject_visits.setdefault(subject_id, []).append({
                "visit": visit_num,
                "folder": folder,
                "files": json_files
            })
        
        # 根據設定選擇訪視
        for subject_id, visits in subject_visits.items():
            # 按訪視次數排序（最新的在前）
            visits.sort(key=lambda x: x["visit"], reverse=True)
            
            # 選擇要使用的訪視
            if self.use_all_visits:
                selected_visits = visits
            else:
                selected_visits = [visits[0]]  # 只用最新的
            
            # 建立 SubjectInfo
            for visit_data in selected_visits:
                subjects.append(
                    SubjectInfo(
                        group=group,
                        id=subject_id,
                        visit=visit_data["visit"],
                        feature_paths=visit_data["files"]
                    )
                )
        
        return subjects
    
    def _is_allowed(self, folder_name: str, group: str) -> bool:
        """檢查資料夾是否在允許清單中"""
        # 如果沒有設定允許清單，允許所有
        if not self.allowed_subjects or group not in self.allowed_subjects:
            return True
        
        allowed_ids = self.allowed_subjects[group]
        if not allowed_ids:
            return True
        
        # 提取基礎ID和完整ID
        base_id, _ = parse_subject_id(folder_name)
        
        # 檢查多種可能的ID格式
        return any([
            folder_name in allowed_ids,
            base_id in allowed_ids,
            str(base_id) in allowed_ids,
            # 提取數字部分再檢查
            str(re.findall(r'\d+', base_id)[0]) in allowed_ids if re.findall(r'\d+', base_id) else False
        ])
    
    def _parse_folder_name(self, folder_name: str, group: str) -> Tuple[int, int]:
        """解析資料夾名稱，提取受試者ID和訪視次數
        
        Returns:
            (subject_id_number, visit_number)
        """
        base_id, visit_number = parse_subject_id(folder_name)
        
        # 提取數字部分作為subject_id
        numbers = re.findall(r'\d+', base_id)
        if numbers:
            subject_id_number = int(numbers[0])
        else:
            # 如果沒有數字，使用hash值
            subject_id_number = hash(base_id) % 100000
        
        return subject_id_number, visit_number
    
    def _collect_json_files(self, folder: Path) -> List[Path]:
        """收集資料夾中的JSON特徵檔案"""
        # 優先使用 LR_difference.json 檔案
        lr_files = sorted(folder.glob("*_LR_difference.json"))
        if lr_files:
            return lr_files
        
        # 否則使用所有JSON檔案
        return sorted(folder.glob("*.json"))
    
    # ==================== 載入特徵 ====================
    def load_features(
        self,
        subjects: List[SubjectInfo],
        embedding_model: str,
        feature_type: str = "difference"
    ) -> List[SubjectFeature]:
        """載入特徵資料
        
        Args:
            subjects: 要載入的個案列表
            embedding_model: 嵌入模型名稱 (vggface, arcface, dlib, deepid, topofr)
            feature_type: 特徵類型 (difference, average, relative)
        
        Returns:
            特徵資料列表
        """
        feature_data_list: List[SubjectFeature] = []
        
        for subject in subjects:
            # 從JSON檔案提取特徵
            features = self._extract_features(
                subject.feature_paths,
                embedding_model,
                feature_type
            )
            
            if features is not None:
                feature_data_list.append(
                    SubjectFeature(
                        subject_info=subject,
                        features={embedding_model: features},
                        feature_type=feature_type
                    )
                )
        
        logger.debug(
            f"載入 {len(feature_data_list)}/{len(subjects)} 個個案的 "
            f"{embedding_model}-{feature_type} 特徵"
        )
        
        return feature_data_list
    
    def _extract_features(
        self,
        json_files: List[Path],
        embedding_model: str,
        feature_type: str
    ) -> Optional[np.ndarray]:
        """從JSON檔案中提取並平均特徵向量"""
        vectors: List[np.ndarray] = []
        
        for json_file in json_files:
            try:
                data = load_json(json_file)
                
                # 根據特徵類型提取對應資料
                if feature_type == "difference":
                    feat = data.get("embedding_differences", {}).get(embedding_model)
                elif feature_type == "average":
                    feat = data.get("embedding_averages", {}).get(embedding_model)
                elif feature_type == "relative":
                    feat = data.get("relative_differences", {}).get(embedding_model)
                else:
                    logger.warning(f"未知的特徵類型: {feature_type}")
                    continue
                
                if feat is not None:
                    vec = np.asarray(feat, dtype=float)
                    vectors.append(vec)
            
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
        stacked = np.vstack(vectors)
        mean_vec = stacked.mean(axis=0)
        
        return mean_vec
    
    # ==================== 公開方法 ====================
    def get_statistics(self, subjects: List[SubjectInfo]) -> Dict:
        """獲取資料集統計資訊"""
        stats = {
            "total_subjects": len(subjects),
            "by_group": {},
            "by_visit": {}
        }
        
        # 按群組統計
        for group in ["P", "ACS", "NAD"]:
            group_subjects = [s for s in subjects if s.group == group]
            stats["by_group"][group] = len(group_subjects)
        
        # 按訪視次數統計
        visit_counts = {}
        for s in subjects:
            visit_counts[s.visit] = visit_counts.get(s.visit, 0) + 1
        stats["by_visit"] = visit_counts
        
        return stats