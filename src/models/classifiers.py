# src/models/classifiers.py
"""分類器工廠模組"""
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class ClassifierType(Enum):
    """支援的分類器類型"""
    RANDOM_FOREST = "Random Forest"
    XGB = "XGBoost"
    SVM = "SVM"
    LOGISTIC = "Logistic Regression"


@dataclass
class ClassifierConfig:
    """分類器配置"""
    classifier_type: ClassifierType
    params: Dict[str, Any]
    need_scaling: bool = False
    random_state: int = 42
    
    @classmethod
    def get_default(cls, classifier_type: ClassifierType) -> 'ClassifierConfig':
        """獲取預設配置"""
        defaults = {
            ClassifierType.RANDOM_FOREST: cls(
                classifier_type=ClassifierType.RANDOM_FOREST,
                params={'n_estimators': 100, 'max_depth': 10},
                need_scaling=False
            ),
            ClassifierType.XGB: cls(
                classifier_type=ClassifierType.XGB,
                params={'n_estimators': 100, 'max_depth': 5, 'n_jobs': -1},
                need_scaling=False
            ),
            ClassifierType.SVM: cls(
                classifier_type=ClassifierType.SVM,
                params={'kernel': 'rbf', 'C': 1.0},
                need_scaling=True
            ),
            ClassifierType.LOGISTIC: cls(
                classifier_type=ClassifierType.LOGISTIC,
                params={'max_iter': 1000, 'solver': 'lbfgs'},
                need_scaling=True
            )
        }
        return defaults[classifier_type]


class ClassifierFactory:
    """分類器工廠
    
    負責創建和管理不同類型的分類器，包含：
    - 統一的創建介面
    - 自動處理資料標準化
    - 參數管理
    """
    
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self._classifiers = {
            ClassifierType.RANDOM_FOREST: RandomForestClassifier,
            ClassifierType.XGB: xgb.XGBClassifier,
            ClassifierType.SVM: SVC,
            ClassifierType.LOGISTIC: LogisticRegression
        }
        
    def create_classifier(
        self, 
        config: Optional[ClassifierConfig] = None,
        classifier_type: Optional[ClassifierType] = None
    ):
        """創建分類器實例
        
        Args:
            config: 分類器配置，若未提供則使用預設
            classifier_type: 分類器類型，當config未提供時使用
            
        Returns:
            分類器實例
        """
        if config is None:
            if classifier_type is None:
                raise ValueError("必須提供 config 或 classifier_type")
            config = ClassifierConfig.get_default(classifier_type)
            
        # 加入 random_state
        params = config.params.copy()
        if 'random_state' not in params:
            params['random_state'] = config.random_state
            
        # 創建分類器
        classifier_class = self._classifiers[config.classifier_type]
        classifier = classifier_class(**params)
        
        logger.info(f"創建分類器: {config.classifier_type.value}")
        return classifier
    
    def prepare_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        config: ClassifierConfig
    ) -> tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
        """準備訓練和測試資料
        
        根據分類器需求決定是否標準化
        
        Args:
            X_train: 訓練資料
            X_test: 測試資料
            config: 分類器配置
            
        Returns:
            處理後的 (X_train, X_test, scaler)
        """
        if config.need_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            logger.debug(f"{config.classifier_type.value} 使用標準化")
            return X_train_scaled, X_test_scaled, scaler
        else:
            return X_train, X_test, None
    
    @staticmethod
    def get_all_types() -> list[ClassifierType]:
        """獲取所有支援的分類器類型"""
        return list(ClassifierType)
    
    @staticmethod
    def from_string(name: str) -> ClassifierType:
        """從字串轉換為分類器類型"""
        for ct in ClassifierType:
            if ct.value == name:
                return ct
        raise ValueError(f"未知的分類器類型: {name}")