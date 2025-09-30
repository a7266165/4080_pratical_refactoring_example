# src/modeltrainer/training_utils.py
"""訓練工具模組 - 整合所有訓練相關功能"""
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Set
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import logging
from src.utils.utils import calculate_metrics

logger = logging.getLogger(__name__)


# ==================== 分類器定義 ====================
class ClassifierType(Enum):
    """支援的分類器類型"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    SVM = "svm"
    LOGISTIC = "logistic"


# 分類器配置
CLASSIFIER_CONFIG = {
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        "needs_scaling": False
    },
    "xgboost": {
        "class": xgb.XGBClassifier,
        "params": {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        },
        "needs_scaling": False
    },
    "svm": {
        "class": SVC,
        "params": {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42,
            'probability': True
        },
        "needs_scaling": True
    },
    "logistic": {
        "class": LogisticRegression,
        "params": {
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        },
        "needs_scaling": True
    }
}


# ==================== 分類器工廠 ====================
def create_classifier(classifier_name: str, custom_params: Optional[Dict] = None):
    """創建分類器實例
    
    Args:
        classifier_name: 分類器名稱
        custom_params: 自定義參數（可選）
    
    Returns:
        分類器實例
    """
    if classifier_name not in CLASSIFIER_CONFIG:
        raise ValueError(f"未知的分類器: {classifier_name}")
    
    config = CLASSIFIER_CONFIG[classifier_name]
    params = custom_params or config['params']
    
    logger.debug(f"創建分類器: {classifier_name}")
    return config['class'](**params)


def prepare_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    classifier_name: str
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """準備訓練和測試資料（根據需要進行標準化）
    
    Args:
        X_train: 訓練特徵
        X_test: 測試特徵
        classifier_name: 分類器名稱
    
    Returns:
        處理後的訓練資料、測試資料、標準化器（如果使用）
    """
    if CLASSIFIER_CONFIG[classifier_name]['needs_scaling']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.debug(f"{classifier_name} 使用標準化")
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train, X_test, None


# ==================== 特徵選擇 ====================
def select_features_by_correlation(
    X: np.ndarray,
    threshold: float = 0.95
) -> Tuple[np.ndarray, List[int]]:
    """基於相關性移除高度相關的特徵
    
    Args:
        X: 特徵矩陣
        threshold: 相關係數閾值
    
    Returns:
        篩選後的特徵矩陣和保留的特徵索引
    """
    n_features = X.shape[1]
    if n_features <= 1:
        return X, list(range(n_features))
    
    # 計算相關矩陣
    corr_matrix = np.corrcoef(X.T)
    
    # 找出要保留的特徵
    keep_features = []
    removed_features: Set[int] = set()
    
    for i in range(corr_matrix.shape[0]):
        if i in removed_features:
            continue
        keep_features.append(i)
        
        # 移除高度相關的特徵
        for j in range(i + 1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > threshold:
                removed_features.add(j)
    
    logger.info(f"相關性過濾: {n_features} -> {len(keep_features)} 個特徵")
    
    return X[:, keep_features], keep_features


def select_features_by_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    importance_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """基於XGBoost特徵重要性選擇特徵
    
    Args:
        X_train: 訓練特徵
        y_train: 訓練標籤
        X_test: 測試特徵
        importance_ratio: 保留的累積重要性比例
    
    Returns:
        篩選後的訓練特徵、測試特徵、選擇的特徵索引
    """
    # 訓練臨時XGBoost模型獲取重要性
    temp_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    temp_model.fit(X_train, y_train)
    
    # 獲取特徵重要性
    importance = temp_model.feature_importances_
    
    # 排序並計算累積重要性
    indices = np.argsort(importance)[::-1]
    cumsum = np.cumsum(importance[indices])
    
    # 找出達到閾值的特徵數量
    if cumsum[-1] > 0:
        n_features = np.searchsorted(cumsum, importance_ratio * cumsum[-1]) + 1
    else:
        n_features = len(indices)
    
    n_features = max(1, min(n_features, len(indices)))
    
    # 選擇最重要的特徵
    selected_features = sorted(indices[:n_features])
    
    logger.info(
        f"XGBoost特徵選擇: {X_train.shape[1]} -> {len(selected_features)} 個特徵"
    )
    
    return (
        X_train[:, selected_features],
        X_test[:, selected_features],
        selected_features
    )


# ==================== 交叉驗證 ====================
class CrossValidator:
    """交叉驗證器"""
    
    def __init__(
        self,
        cv_method: str = "5-fold",
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            cv_method: 交叉驗證方法 ("5-fold" 或 "loso")
            n_folds: 折數
            random_state: 隨機種子
        """
        self.cv_method = cv_method.lower()
        self.n_folds = n_folds
        self.random_state = random_state
    
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_name: str,
        use_feature_selection: bool = False
    ) -> Dict:
        """執行交叉驗證
        
        Args:
            X: 特徵矩陣
            y: 標籤
            subject_ids: 受試者ID列表
            classifier_name: 分類器名稱
            use_feature_selection: 是否使用特徵選擇
        
        Returns:
            評估指標字典
        """
        if "loso" in self.cv_method:
            return self._loso_cv(
                X, y, subject_ids, classifier_name, use_feature_selection
            )
        else:
            return self._kfold_cv(
                X, y, subject_ids, classifier_name, use_feature_selection
            )
    
    def _loso_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_name: str,
        use_feature_selection: bool
    ) -> Dict:
        """Leave-One-Subject-Out 交叉驗證"""
        unique_subjects = list(set(subject_ids))
        n_subjects = len(unique_subjects)
        
        logger.info(f"執行 LOSO 交叉驗證，共 {n_subjects} 個受試者")
        
        all_y_true = []
        all_y_pred = []
        
        for i, test_subject in enumerate(unique_subjects):
            # 分割資料
            test_idx = [j for j, sid in enumerate(subject_ids) if sid == test_subject]
            train_idx = [j for j, sid in enumerate(subject_ids) if sid != test_subject]
            
            if not test_idx or not train_idx:
                continue
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 特徵選擇
            if use_feature_selection and classifier_name == "xgboost":
                X_train, X_test, _ = select_features_by_importance(
                    X_train, y_train, X_test
                )
            elif use_feature_selection:
                selected_features = select_features_by_correlation(X_train)[1]
                X_train = X_train[:, selected_features]
                X_test = X_test[:, selected_features]
            
            # 資料預處理
            X_train, X_test, _ = prepare_data(X_train, X_test, classifier_name)
            
            # 訓練和預測
            classifier = create_classifier(classifier_name)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            
            # 多數決（如果有多個樣本）
            y_pred_final = 1 if np.mean(y_pred) >= 0.5 else 0
            y_true_final = 1 if np.mean(y_test) >= 0.5 else 0
            
            all_y_true.append(y_true_final)
            all_y_pred.append(y_pred_final)
            
            if (i + 1) % 20 == 0:
                logger.debug(f"  進度: {i + 1}/{n_subjects}")
        
        return calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
    
    def _kfold_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_name: str,
        use_feature_selection: bool
    ) -> Dict:
        """K-Fold 交叉驗證（按受試者分組）"""
        logger.info(f"執行 {self.n_folds}-fold 交叉驗證")
        
        # 建立受試者級別的資料
        unique_subjects = list(set(subject_ids))
        subject_to_indices = {subj: [] for subj in unique_subjects}
        for idx, subj in enumerate(subject_ids):
            subject_to_indices[subj].append(idx)
        
        # 受試者級別的標籤
        subject_labels = []
        for subj in unique_subjects:
            indices = subject_to_indices[subj]
            labels = [y[i] for i in indices]
            subject_labels.append(1 if np.mean(labels) >= 0.5 else 0)
        
        subject_labels = np.array(subject_labels)
        
        # 分層K折
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        all_y_true = []
        all_y_pred = []
        
        for fold_idx, (train_subj_idx, test_subj_idx) in enumerate(
            skf.split(unique_subjects, subject_labels)
        ):
            # 獲取訓練和測試索引
            train_idx = []
            test_idx = []
            test_subjects = [unique_subjects[i] for i in test_subj_idx]
            
            for i in train_subj_idx:
                train_idx.extend(subject_to_indices[unique_subjects[i]])
            for i in test_subj_idx:
                test_idx.extend(subject_to_indices[unique_subjects[i]])
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 特徵選擇
            if use_feature_selection and classifier_name == "xgboost":
                X_train, X_test, _ = select_features_by_importance(
                    X_train, y_train, X_test
                )
            elif use_feature_selection:
                selected_features = select_features_by_correlation(X_train)[1]
                X_train = X_train[:, selected_features]
                X_test = X_test[:, selected_features]
            
            # 資料預處理
            X_train, X_test, _ = prepare_data(X_train, X_test, classifier_name)
            
            # 訓練分類器
            classifier = create_classifier(classifier_name)
            classifier.fit(X_train, y_train)
            
            # 對每個測試受試者預測（多數決）
            for test_subj in test_subjects:
                subj_test_idx = [
                    test_idx.index(i)
                    for i in test_idx
                    if subject_ids[i] == test_subj
                ]
                
                y_pred_subj = classifier.predict(X_test[subj_test_idx])
                y_true_subj = y_test[subj_test_idx]
                
                y_pred_final = 1 if np.mean(y_pred_subj) >= 0.5 else 0
                y_true_final = 1 if np.mean(y_true_subj) >= 0.5 else 0
                
                all_y_true.append(y_true_final)
                all_y_pred.append(y_pred_final)
        
        return calculate_metrics(np.array(all_y_true), np.array(all_y_pred))