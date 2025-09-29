# src/modeltrainer/trainfactory/cross_validation.py
"""交叉驗證策略模組"""
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .classifier_factory import ClassifierFactory
from .feature_selection import FeatureSelector
import logging
from src.utils.utils import calculate_metrics

logger = logging.getLogger(__name__)


class CrossValidator:
    """交叉驗證器"""

    def __init__(
        self, cv_method: str = "5-Fold", n_folds: int = 5, random_state: int = 42
    ):
        self.cv_method = cv_method
        self.n_folds = n_folds
        self.random_state = random_state

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_name: str,
        use_feature_selection: bool = False,
        feature_selection_method: str = "correlation",
    ) -> Dict:
        """執行交叉驗證"""

        if "LOSO" in self.cv_method.upper():
            return self._loso_cv(
                X,
                y,
                subject_ids,
                classifier_name,
                use_feature_selection,
                feature_selection_method,
            )
        else:
            return self._kfold_cv(
                X,
                y,
                subject_ids,
                classifier_name,
                use_feature_selection,
                feature_selection_method,
            )

    def _loso_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_name: str,
        use_feature_selection: bool,
        feature_selection_method: str,
    ) -> Dict:
        """Leave-One-Subject-Out 交叉驗證"""
        unique_subjects = list(set(subject_ids))
        n_subjects = len(unique_subjects)

        logger.info(f"執行 LOSO 交叉驗證，共 {n_subjects} 個獨立受試者")

        all_y_true = []
        all_y_pred = []

        for i, test_subject in enumerate(unique_subjects):
            # 建立訓練和測試索引
            test_indices = [
                idx for idx, sid in enumerate(subject_ids) if sid == test_subject
            ]
            train_indices = [
                idx for idx, sid in enumerate(subject_ids) if sid != test_subject
            ]

            if not test_indices or not train_indices:
                continue

            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

            # 特徵選擇
            if use_feature_selection:
                X_train, X_test = self._apply_feature_selection(
                    X_train, y_train, X_test, classifier_name, feature_selection_method
                )

            # 資料預處理
            X_train, X_test, _ = ClassifierFactory.prepare_data(
                X_train, X_test, classifier_name
            )

            # 訓練分類器
            classifier = ClassifierFactory.create_classifier(classifier_name)
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

            if (i + 1) % 20 == 0:
                logger.debug(f"  進度: {i + 1}/{n_subjects}")

        return calculate_metrics(np.array(all_y_true), np.array(all_y_pred))

    def _kfold_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_name: str,
        use_feature_selection: bool,
        feature_selection_method: str,
    ) -> Dict:
        """K-Fold 交叉驗證（考慮受試者分組）"""
        logger.info(f"執行 {self.n_folds}-fold 交叉驗證")

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
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_subject_idx, test_subject_idx) in enumerate(
            skf.split(unique_subjects, subject_labels)
        ):
            logger.debug(f"  Fold {fold_idx + 1}/{self.n_folds}")

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
            if use_feature_selection:
                X_train, X_test = self._apply_feature_selection(
                    X_train, y_train, X_test, classifier_name, feature_selection_method
                )

            # 資料預處理
            X_train, X_test, _ = ClassifierFactory.prepare_data(
                X_train, X_test, classifier_name
            )

            # 訓練分類器
            classifier = ClassifierFactory.create_classifier(classifier_name)
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

        return calculate_metrics(np.array(all_y_true), np.array(all_y_pred))

    def _apply_feature_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        classifier_name: str,
        method: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """應用特徵選擇"""
        if classifier_name in ["SVM", "Logistic Regression"] or method == "correlation":
            X_train, X_test, _ = FeatureSelector.select_by_correlation(
                X_train, X_test, threshold=0.95
            )
        elif classifier_name == "XGBoost" or method == "xgb_importance":
            X_train, X_test, _ = FeatureSelector.select_by_xgb_importance(
                X_train, y_train, X_test, importance_ratio=0.8
            )

        return X_train, X_test
