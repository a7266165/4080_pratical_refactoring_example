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


# ========== JSON 序列化輔助類 ==========
class NumpyEncoder(json.JSONEncoder):
    """處理 NumPy 類型的 JSON 編碼器"""
    def default(self, obj):
        # 處理 NumPy 整數類型
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp,
                          np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        # 處理 NumPy 浮點數類型
        elif isinstance(obj, (np.floating, np.float_, np.float16,
                            np.float32, np.float64)):
            return float(obj)
        
        # 處理 NumPy 布林類型
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        
        # 處理 NumPy 陣列
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # 處理 pandas 類型
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        
        # 處理 bytes 類型
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        
        # 其他類型交給預設處理
        return super().default(obj)


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
    """儲存JSON檔案（支援 NumPy 類型）
    
    Args:
        data: 要儲存的資料
        filepath: 檔案路徑
        indent: 縮排空格數
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用自定義編碼器處理 NumPy 類型
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, cls=NumpyEncoder)


def convert_to_serializable(obj: Any) -> Any:
    """遞迴轉換物件為可序列化的格式
    
    這是一個備用方法，可以在需要時手動轉換資料
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        return obj


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
    # 確保輸入是 numpy 陣列
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 基本指標
    cm = confusion_matrix(y_true, y_pred)
    
    # 處理混淆矩陣（確保是 2x2）
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # 如果只有一個類別，填充為 2x2
        tn = fp = fn = tp = 0
        if cm.shape == (1, 1):
            if y_true[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]
    
    # 計算各項指標（轉換為 Python 原生類型）
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),  # 轉換為列表
    }
    
    # 計算 AUC (如果有機率值)
    if include_auc and y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            logging.warning(f"無法計算AUC: {e}")
            metrics["auc"] = None
    
    return metrics


def calculate_dataset_stats(
    X: np.ndarray, y: np.ndarray, subject_ids: List[str]
) -> Dict[str, Any]:
    """計算資料集統計資訊"""
    unique_subjects = len(set(subject_ids))
    
    # 確保回傳的都是 Python 原生類型
    return {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_subjects": int(unique_subjects),
        "n_health": int(np.sum(y == 0)),
        "n_patient": int(np.sum(y == 1)),
        "samples_per_subject": float(len(X) / unique_subjects) if unique_subjects > 0 else 0.0,
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


def ensure_python_types(data: Any) -> Any:
    """確保資料中的所有值都是 Python 原生類型
    
    這個函數可以在儲存 JSON 前使用，確保不會有序列化問題
    """
    if isinstance(data, dict):
        return {k: ensure_python_types(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [ensure_python_types(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.bool_, np.bool8)):
        return bool(data)
    elif pd.api.types.is_numeric_dtype(type(data)):
        # 處理 pandas 數值類型
        return float(data) if pd.api.types.is_float_dtype(type(data)) else int(data)
    else:
        return data