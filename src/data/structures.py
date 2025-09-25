# src/data/structures.py
"""資料結構定義"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

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

    def __str__(self) -> str: # debug用
        return f"{self.group}{self.id}-{self.visit}"

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
    groups: Dict[str, int]  # {'ACS': n, 'NAD': n, 'P': n}
    feature_dim: int