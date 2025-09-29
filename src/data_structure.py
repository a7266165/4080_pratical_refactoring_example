# src/data_structures.py
"""統一的資料結構定義"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

@dataclass
class SubjectInfo:
    """個案資訊"""
    group: str              # "ACS", "NAD", "P"
    id: int                 # 1, 2, 3, ...
    visit: int              # 1, 2, 3, ...
    feature_paths: List[Path]
    
    @property
    def label(self) -> int:
        """模型訓練標籤"""
        return 1 if self.group == "P" else 0
    
    @property
    def subject_id(self) -> str:
        """個案編號"""
        return f"{self.group}{self.id}"

@dataclass
class SubjectFeature:
    """特徵資料"""
    subject_info: SubjectInfo
    features: Dict[str, np.ndarray]  # {model_name: features}
    feature_type: str  # 'difference', 'average', 'relative'

@dataclass
class DatasetInfo:
    """資料集統計資訊"""
    n_samples: int
    n_subjects: int
    n_health: int
    n_patient: int
    groups: Dict[str, int]
    feature_dim: int