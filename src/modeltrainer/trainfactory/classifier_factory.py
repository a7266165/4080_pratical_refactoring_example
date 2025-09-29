# src/modeltrainer/trainfactory/classifier_factory.py
"""分類器工廠模組"""
from enum import Enum
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class ClassifierFactory:
    """分類器工廠"""
    
    # 分類器映射
    CLASSIFIERS = {
        "Random Forest": RandomForestClassifier,
        "XGBoost": xgb.XGBClassifier,
        "SVM": SVC,
        "Logistic Regression": LogisticRegression
    }
    
    # 預設參數
    DEFAULT_PARAMS = {
        "Random Forest": {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        "XGBoost": {
            'n_estimators': 100,
            'max_depth': 5,
            'n_jobs': -1,
            'random_state': 42
        },
        "SVM": {
            'kernel': 'rbf',
            'C': 1.0,
            'random_state': 42
        },
        "Logistic Regression": {
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42
        }
    }
    
    # 是否需要標準化
    NEED_SCALING = {
        "Random Forest": False,
        "XGBoost": False,
        "SVM": True,
        "Logistic Regression": True
    }
    
    @classmethod
    def create_classifier(cls, classifier_name: str, params: Optional[Dict] = None):
        """創建分類器實例"""
        if classifier_name not in cls.CLASSIFIERS:
            raise ValueError(f"未知的分類器: {classifier_name}")
        
        # 使用預設參數或自定義參數
        if params is None:
            params = cls.DEFAULT_PARAMS[classifier_name]
        
        classifier_class = cls.CLASSIFIERS[classifier_name]
        classifier = classifier_class(**params)
        
        logger.info(f"創建分類器: {classifier_name}")
        return classifier
    
    @classmethod
    def prepare_data(
        cls,
        X_train: np.ndarray,
        X_test: np.ndarray,
        classifier_name: str
    ) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
        """準備訓練和測試資料"""
        if cls.NEED_SCALING.get(classifier_name, False):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            logger.debug(f"{classifier_name} 使用標準化")
            return X_train_scaled, X_test_scaled, scaler
        else:
            return X_train, X_test, None