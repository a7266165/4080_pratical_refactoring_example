# src/models/cross_validation.py
"""交叉驗證策略模組"""
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef

from src.models.classifiers import ClassifierFactory, ClassifierConfig, ClassifierType
from src.features.selection import FeatureSelector, SelectionConfig
import logging

logger = logging.getLogger(__name__)


class CVMethod(Enum):
    """交叉驗證方法"""
    LOSO = "LOSO"  # Leave-One-Subject-Out
    KFOLD = "K-Fold"


@dataclass
class CVConfig:
    """交叉驗證配置"""
    method: CVMethod = CVMethod.LOSO
    n_folds: int = 5  # 僅用於 K-Fold
    random_state: int = 42
    feature_selection: Optional[SelectionConfig] = None


@dataclass
class CVResults:
    """交叉驗證結果"""
    confusion_matrix: np.ndarray
    accuracy: float
    mcc: float
    sensitivity: float
    specificity: float
    y_true: np.ndarray
    y_pred: np.ndarray
    test_subjects: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            'confusion_matrix': self.confusion_matrix.tolist(),
            'accuracy': float(self.accuracy),
            'mcc': float(self.mcc),
            'sensitivity': float(self.sensitivity),
            'specificity': float(self.specificity)
        }
    
    def print_summary(self):
        """印出結果摘要"""
        cm = self.confusion_matrix
        print(f"    準確率: {self.accuracy:.4f}")
        print(f"    MCC: {self.mcc:.4f}")
        print(f"    靈敏度: {self.sensitivity:.4f}")
        print(f"    特異度: {self.specificity:.4f}")
        print(f"    混淆矩陣:")
        print(f"      TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"      FN={cm[1,0]}, TP={cm[1,1]}")


class CrossValidator:
    """交叉驗證器
    
    統一的交叉驗證介面，支援：
    - Leave-One-Subject-Out (LOSO)
    - K-Fold 交叉驗證
    - 整合特徵選擇
    - 自動處理受試者層級的劃分
    """
    
    def __init__(self, config: Optional[CVConfig] = None):
        """
        Args:
            config: 交叉驗證配置
        """
        self.config = config or CVConfig()
        self.classifier_factory = ClassifierFactory()
        
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_config: ClassifierConfig
    ) -> CVResults:
        """執行交叉驗證
        
        Args:
            X: 特徵矩陣
            y: 標籤
            subject_ids: 受試者ID列表
            classifier_config: 分類器配置
            
        Returns:
            交叉驗證結果
        """
        if self.config.method == CVMethod.LOSO:
            return self._loso_cv(X, y, subject_ids, classifier_config)
        elif self.config.method == CVMethod.KFOLD:
            return self._kfold_cv(X, y, subject_ids, classifier_config)
        else:
            raise ValueError(f"不支援的CV方法: {self.config.method}")
    
    def _loso_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_config: ClassifierConfig
    ) -> CVResults:
        """Leave-One-Subject-Out 交叉驗證"""
        unique_subjects = list(set(subject_ids))
        n_subjects = len(unique_subjects)
        
        logger.info(f"執行 LOSO 交叉驗證，共 {n_subjects} 個獨立受試者")
        
        all_y_true = []
        all_y_pred = []
        all_test_subjects = []
        
        for i, test_subject in enumerate(unique_subjects):
            # 建立訓練和測試索引
            test_indices = [idx for idx, sid in enumerate(subject_ids) if sid == test_subject]
            train_indices = [idx for idx, sid in enumerate(subject_ids) if sid != test_subject]
            
            if not test_indices or not train_indices:
                continue
                
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            # 特徵選擇（如果配置了）
            if self.config.feature_selection:
                selector = FeatureSelector(self.config.feature_selection)
                X_train, X_test, _ = selector.select_features(X_train, y_train, X_test)
            
            # 資料預處理
            X_train, X_test, _ = self.classifier_factory.prepare_data(
                X_train, X_test, classifier_config
            )
            
            # 訓練分類器
            classifier = self.classifier_factory.create_classifier(classifier_config)
            classifier.fit(X_train, y_train)
            
            # 預測
            y_pred = classifier.predict(X_test)
            
            # 如果該受試者有多個樣本，取多數決
            if len(y_pred) > 1:
                y_pred_final = 1 if np.mean(y_pred) >= 0.5 else 0
                y_true_final = 1 if np.mean(y_test) >= 0.5 else 0
            else:
                y_pred_final = y_pred[0]
                y_true_final = y_test[0]
            
            all_y_true.append(y_true_final)
            all_y_pred.append(y_pred_final)
            all_test_subjects.append(test_subject)
            
            if (i + 1) % 20 == 0:
                logger.debug(f"  進度: {i + 1}/{n_subjects}")
        
        return self._calculate_metrics(
            np.array(all_y_true),
            np.array(all_y_pred),
            all_test_subjects
        )
    
    def _kfold_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_config: ClassifierConfig
    ) -> CVResults:
        """K-Fold 交叉驗證（考慮受試者分組）"""
        logger.info(f"執行 {self.config.n_folds}-fold 交叉驗證")
        
        # 建立受試者到索引的映射
        unique_subjects = list(set(subject_ids))
        subject_to_indices = {subj: [] for subj in unique_subjects}
        for idx, subj in enumerate(subject_ids):
            subject_to_indices[subj].append(idx)
        
        # 建立受試者級別的標籤
        subject_labels = []
        for subj in unique_subjects:
            indices = subject_to_indices[subj]
            labels = [y[i] for i in indices]
            subject_label = 1 if np.mean(labels) >= 0.5 else 0
            subject_labels.append(subject_label)
        
        subject_labels = np.array(subject_labels)
        
        # 使用StratifiedKFold對受試者進行分組
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        all_y_true = []
        all_y_pred = []
        
        for fold_idx, (train_subject_idx, test_subject_idx) in enumerate(skf.split(unique_subjects, subject_labels)):
            logger.debug(f"  Fold {fold_idx + 1}/{self.config.n_folds}")
            
            # 獲取訓練和測試受試者
            train_subjects = [unique_subjects[i] for i in train_subject_idx]
            test_subjects = [unique_subjects[i] for i in test_subject_idx]
            
            # 獲取對應的樣本索引
            train_indices = []
            test_indices = []
            for subj in train_subjects:
                train_indices.extend(subject_to_indices[subj])
            for subj in test_subjects:
                test_indices.extend(subject_to_indices[subj])
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            # 特徵選擇
            if self.config.feature_selection:
                selector = FeatureSelector(self.config.feature_selection)
                X_train, X_test, _ = selector.select_features(X_train, y_train, X_test)
            
            # 資料預處理
            X_train, X_test, _ = self.classifier_factory.prepare_data(
                X_train, X_test, classifier_config
            )
            
            # 訓練分類器
            classifier = self.classifier_factory.create_classifier(classifier_config)
            classifier.fit(X_train, y_train)
            
            # 預測（對每個測試受試者進行預測）
            for test_subj in test_subjects:
                subj_indices = [i for i in test_indices if subject_ids[i] == test_subj]
                subj_test_idx = [test_indices.index(i) for i in subj_indices]
                
                y_pred_subj = classifier.predict(X_test[subj_test_idx])
                y_true_subj = y_test[subj_test_idx]
                
                # 取多數決
                if len(y_pred_subj) > 1:
                    y_pred_final = 1 if np.mean(y_pred_subj) >= 0.5 else 0
                    y_true_final = 1 if np.mean(y_true_subj) >= 0.5 else 0
                else:
                    y_pred_final = y_pred_subj[0]
                    y_true_final = y_true_subj[0]
                
                all_y_true.append(y_true_final)
                all_y_pred.append(y_pred_final)
        
        return self._calculate_metrics(
            np.array(all_y_true),
            np.array(all_y_pred)
        )
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_subjects: Optional[List[str]] = None
    ) -> CVResults:
        """計算評估指標"""
        cm = confusion_matrix(y_true, y_pred)
        
        # 計算各項指標
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # 靈敏度和特異度
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return CVResults(
            confusion_matrix=cm,
            accuracy=acc,
            mcc=mcc,
            sensitivity=sensitivity,
            specificity=specificity,
            y_true=y_true,
            y_pred=y_pred,
            test_subjects=test_subjects
        )