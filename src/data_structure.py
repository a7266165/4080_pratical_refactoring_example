# src/data_structures.py
"""統一的資料結構定義"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class Picture:
    """圖片資料"""

    path: str
    image: Optional[np.ndarray] = None  # (H, W, 3) RGB array
    subject_id: str = ""
    group: str = ""  # ACS/NAD/P
    visit: int = 1


@dataclass
class ExtractedFeatures:
    """萃取的特徵（步驟0-5的輸出）"""

    # 這其實就是現有的 *_LR_difference.json 內容
    subject_id: str
    group: str
    visit: int
    embedding_differences: Dict[str, np.ndarray]  # {model: features}
    embedding_averages: Dict[str, np.ndarray]
    relative_differences: Dict[str, np.ndarray]


@dataclass
class TrainingDataset:
    """訓練資料集"""

    X: np.ndarray  # (n_samples, n_features)
    y: np.ndarray  # (n_samples,)
    subject_ids: List[str]
    metadata: Dict[str, Any]  # 包含使用的模型、特徵類型等資訊


@dataclass
class ModelResults:
    """模型訓練結果"""

    classifier: str
    cv_method: str
    metrics: Dict[str, float]  # accuracy, mcc, sensitivity, specificity
    confusion_matrix: np.ndarray
