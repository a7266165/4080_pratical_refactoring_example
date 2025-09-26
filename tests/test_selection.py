# tests/test_features/test_selection.py
import pytest
import numpy as np
from src.features.selection import FeatureSelector, SelectionConfig, SelectionMethod

class TestFeatureSelector:
    
    def test_correlation_filtering(self):
        """測試相關性過濾"""
        # 建立高度相關的特徵
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[:, 1] = X[:, 0] * 0.99 + np.random.randn(100) * 0.01  # 高度相關
        
        selector = FeatureSelector(
            SelectionConfig(
                method=SelectionMethod.CORRELATION,
                correlation_threshold=0.95
            )
        )
        
        X_selected, _, indices = selector.select_features(X, None)
        
        # 應該移除高度相關的特徵
        assert X_selected.shape[1] < X.shape[1]
        assert 0 in indices  # 保留第一個
        assert 1 not in indices  # 移除相關的
    
    def test_xgb_importance(self):
        """測試XGBoost重要性篩選"""
        # 建立資料：前兩個特徵是重要的
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 只依賴前兩個特徵
        
        selector = FeatureSelector(
            SelectionConfig(
                method=SelectionMethod.XGB_IMPORTANCE,
                importance_ratio=0.8
            )
        )
        
        X_selected, _, indices = selector.select_features(X, y)
        
        # 應該選出重要的特徵
        assert 0 in indices
        assert 1 in indices
        assert X_selected.shape[1] < X.shape[1]