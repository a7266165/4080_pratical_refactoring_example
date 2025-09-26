# src/features/selection.py
"""特徵選擇模組"""
from enum import Enum
from typing import Union, Tuple, Optional, List, Set, Dict
import numpy as np
import xgboost as xgb
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """特徵選擇方法"""
    CORRELATION = "correlation"
    XGB_IMPORTANCE = "xgb_importance"
    BOTH = "both"
    NONE = "none"


@dataclass
class SelectionConfig:
    """特徵選擇配置"""
    method: SelectionMethod = SelectionMethod.CORRELATION
    correlation_threshold: float = 0.9
    importance_ratio: float = 0.8
    xgb_params: Dict = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42,
                'n_jobs': -1
            }


class FeatureSelector:
    """特徵選擇器
    
    支援多種特徵選擇策略：
    - 相關性過濾：移除高度相關的特徵
    - XGBoost重要性：基於樹模型的特徵重要性
    - 組合策略：依序應用多種方法
    """
    
    # 預設的模型特徵選擇策略
    MODEL_STRATEGIES = {
        'XGBoost': SelectionMethod.XGB_IMPORTANCE,
        'Random Forest': SelectionMethod.XGB_IMPORTANCE,
        'SVM': SelectionMethod.CORRELATION,
        'Logistic Regression': SelectionMethod.CORRELATION
    }
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        """
        Args:
            config: 特徵選擇配置，若未提供則使用預設值
        """
        self.config = config or SelectionConfig()
        self.selected_indices: Optional[List[int]] = None
        self.feature_importance: Optional[np.ndarray] = None
        
    def select_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        method: Optional[Union[SelectionMethod, str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        """統一的特徵選擇介面
        
        Args:
            X_train: 訓練集特徵
            y_train: 訓練集標籤
            X_test: 測試集特徵（可選）
            method: 選擇方法，若未指定則使用config中的設定
            
        Returns:
            - X_train_selected: 篩選後的訓練集
            - X_test_selected: 篩選後的測試集 (如果有提供)
            - selected_indices: 被選中的特徵索引
        """
        if method is None:
            method = self.config.method
        elif isinstance(method, str):
            method = SelectionMethod(method)
            
        # 特殊情況：無需選擇
        if method == SelectionMethod.NONE or X_train.shape[1] <= 1:
            all_indices = list(range(X_train.shape[1]))
            return X_train, X_test, all_indices
            
        # 執行特徵選擇
        if method == SelectionMethod.CORRELATION:
            return self._select_by_correlation(X_train, X_test)
        elif method == SelectionMethod.XGB_IMPORTANCE:
            return self._select_by_importance(X_train, y_train, X_test)
        elif method == SelectionMethod.BOTH:
            # 先相關性過濾，再重要性篩選
            X_train, X_test, indices1 = self._select_by_correlation(X_train, X_test)
            if len(indices1) > 1:  # 只有還有多個特徵時才繼續篩選
                X_train, X_test, indices2 = self._select_by_importance(X_train, y_train, X_test)
                # 組合索引映射
                final_indices = [indices1[i] for i in indices2]
            else:
                final_indices = indices1
            return X_train, X_test, final_indices
        else:
            raise ValueError(f"不支援的特徵選擇方法: {method}")
    
    def _select_by_correlation(
        self, 
        X_train: np.ndarray, 
        X_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        """基於相關性的特徵選擇
        
        移除高度相關的特徵（相關係數 > threshold）
        """
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
                if abs(corr_matrix[i, j]) > self.config.correlation_threshold:
                    removed_features.add(j)
        
        logger.info(
            f"相關性過濾: {n_features} -> {len(keep_features)} 個特徵 "
            f"(移除 {len(removed_features)} 個, 閾值={self.config.correlation_threshold})"
        )
        
        # 應用選擇
        X_train_selected = X_train[:, keep_features]
        X_test_selected = X_test[:, keep_features] if X_test is not None else None
        
        self.selected_indices = keep_features
        return X_train_selected, X_test_selected, keep_features
    
    def _select_by_importance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        """基於XGBoost特徵重要性的選擇
        
        保留累積重要性達到importance_ratio的特徵
        """
        # 訓練XGBoost模型
        xgb_model = xgb.XGBClassifier(**self.config.xgb_params)
        xgb_model.fit(X_train, y_train)
        
        # 獲取特徵重要性
        importance = xgb_model.feature_importances_
        self.feature_importance = importance
        
        # 排序並計算累積重要性
        indices = np.argsort(importance)[::-1]
        cumsum = np.cumsum(importance[indices])
        
        # 找出累積重要性達到閾值的特徵數量
        if cumsum[-1] > 0:  # 避免除零
            n_features = np.searchsorted(
                cumsum, 
                self.config.importance_ratio * cumsum[-1]
            ) + 1
        else:
            n_features = len(indices)
            
        n_features = max(1, min(n_features, len(indices)))
        
        # 選擇最重要的特徵
        selected_features = sorted(indices[:n_features])  # 保持原始順序
        
        logger.info(
            f"XGBoost特徵選擇: {X_train.shape[1]} -> {len(selected_features)} 個特徵 "
            f"(累積重要性 {self.config.importance_ratio*100:.0f}%)"
        )
        
        # 應用選擇
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features] if X_test is not None else None
        
        self.selected_indices = selected_features
        return X_train_selected, X_test_selected, selected_features
    
    @classmethod
    def get_method_for_classifier(cls, classifier_name: str) -> SelectionMethod:
        """獲取分類器建議的特徵選擇方法
        
        Args:
            classifier_name: 分類器名稱
            
        Returns:
            建議的特徵選擇方法
        """
        return cls.MODEL_STRATEGIES.get(classifier_name, SelectionMethod.NONE)