"""工具函數模組"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)


# ========== ID 處理 ==========
def parse_subject_id(subject_id: str) -> Tuple[str, int]:
    """解析個案ID，分離基礎ID和訪視次數
    Examples:
        "P15-2" -> ("P15", 2)
        "ACS1" -> ("ACS1", 1)
    """
    if "-" in subject_id:
        parts = subject_id.split("-")
        return parts[0], int(parts[1])
    return subject_id, 1


# ========== 檔案 I/O ==========
def load_json(filepath: Path) -> Dict[str, Any]:
    """載入JSON檔案"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Path, indent: int = 2):
    """儲存JSON檔案"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ========== 資料驗證 ==========
def validate_data_consistency(
    X: np.ndarray, y: np.ndarray, subject_ids: List[str]
) -> bool:
    """驗證資料一致性"""
    n_samples = X.shape[0]
    if len(y) != n_samples or len(subject_ids) != n_samples:
        logging.error(
            f"資料維度不一致: X={n_samples}, y={len(y)}, ids={len(subject_ids)}"
        )
        return False
    if not np.all(np.isfinite(X)):
        logging.warning("特徵包含 NaN 或無限值")
        return False
    return True


# ========== 計算指標 ==========
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    include_auc: bool = True,
) -> Dict[str, Any]:
    """計算完整的分類指標

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        y_prob: 預測機率 (用於計算AUC，可選)
        include_auc: 是否計算AUC (需要y_prob)

    Returns:
        包含各種指標的字典：
        - accuracy: 準確率
        - precision: 精確率
        - recall: 召回率 (敏感度)
        - f1: F1分數
        - mcc: Matthews相關係數
        - sensitivity: 敏感度 (=recall)
        - specificity: 特異度
        - confusion_matrix: 混淆矩陣
        - auc: ROC-AUC (如果提供y_prob)
    """
    # 基本指標
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 計算各項指標
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,  # = recall
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "confusion_matrix": cm.tolist(),
    }

    # 計算 AUC (如果有機率值)
    if include_auc and y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logging.warning(f"無法計算AUC: {e}")
            metrics["auc"] = None

    return metrics


def calculate_dataset_stats(
    X: np.ndarray, y: np.ndarray, subject_ids: List[str]
) -> Dict[str, Any]:
    """計算資料集統計資訊"""
    unique_subjects = len(set(subject_ids))
    return {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_subjects": unique_subjects,
        "n_health": np.sum(y == 0),
        "n_patient": np.sum(y == 1),
        "samples_per_subject": len(X) / unique_subjects if unique_subjects > 0 else 0,
    }


# ========== 其他常用功能 ==========
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """設定日誌記錄器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
