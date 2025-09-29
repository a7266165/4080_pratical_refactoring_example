# src/modeltrainer/trainfactory/feature_selection.py
"""特徵選擇模組"""
from typing import Tuple, List, Optional, Set
import numpy as np
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class FeatureSelector:
    """特徵選擇器"""
    
    @staticmethod
    def select_by_correlation(
        X_train: np.ndarray, 
        X_test: np.ndarray,
        threshold: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """基於相關性的特徵選擇"""
        n_features = X_train.shape[1]
        if n_features <= 1:
            return X_train, X_test, list(range(n_features))
        
        # 計算相關矩陣
        corr_matrix = np.corrcoef(X_train.T)
        
        # 找出要保留的特徵
        keep_features = []
        removed_features: Set[int] = set()
        
        for i in range(corr_matrix.shape[0]):
            if i in removed_features:
                continue
            keep_features.append(i)
            
            # 標記高度相關的特徵
            for j in range(i + 1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > threshold:
                    removed_features.add(j)
        
        logger.info(
            f"相關性過濾: {n_features} -> {len(keep_features)} 個特徵 "
            f"(移除 {len(removed_features)} 個, 閾值={threshold})"
        )
        
        # 應用選擇
        X_train_selected = X_train[:, keep_features]
        X_test_selected = X_test[:, keep_features]
        
        return X_train_selected, X_test_selected, keep_features
    
    @staticmethod
    def select_by_xgb_importance(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        importance_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """基於XGBoost特徵重要性的選擇"""
        # 訓練XGBoost模型
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        # 獲取特徵重要性
        importance = xgb_model.feature_importances_
        
        # 排序並計算累積重要性
        indices = np.argsort(importance)[::-1]
        cumsum = np.cumsum(importance[indices])
        
        # 找出累積重要性達到閾值的特徵數量
        if cumsum[-1] > 0:
            n_features = np.searchsorted(
                cumsum, 
                importance_ratio * cumsum[-1]
            ) + 1
        else:
            n_features = len(indices)
            
        n_features = max(1, min(n_features, len(indices)))
        
        # 選擇最重要的特徵
        selected_features = sorted(indices[:n_features])
        
        logger.info(
            f"XGBoost特徵選擇: {X_train.shape[1]} -> {len(selected_features)} 個特徵 "
            f"(累積重要性 {importance_ratio*100:.0f}%)"
        )
        
        # 應用選擇
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        
        return X_train_selected, X_test_selected, selected_features 