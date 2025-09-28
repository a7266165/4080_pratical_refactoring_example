# src/modeltrainer/modeltrainer.py
"""模型訓練管線"""
import logging
from typing import Dict, List, Any
import sys
from pathlib import Path

# 暫時重用 legacy_V2
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "legacy_V2"))

from src.models.classifiers import ClassifierFactory, ClassifierConfig, ClassifierType
from src.train.cross_validation import CrossValidator, CVConfig, CVMethod
from src.features.selection import FeatureSelector, SelectionConfig, SelectionMethod

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型訓練器

    負責：
    1. 執行不同的交叉驗證策略
    2. 訓練多種分類器
    3. 特徵選擇
    4. 返回訓練結果
    """

    def __init__(self, cv_methods: List[str], classifiers: List[ClassifierType]):
        self.cv_methods = cv_methods
        self.classifiers = classifiers

    def train(self, datasets: Dict) -> Dict:
        """訓練所有配置組合

        Args:
            datasets: FeatureLoader 輸出的資料集字典
                     {embedding_feature: {'X': ..., 'y': ..., 'subject_ids': ...}}

        Returns:
            所有訓練結果
        """
        logger.info("開始訓練模型")
        all_results = {}

        # 對每個資料集
        for dataset_name, dataset in datasets.items():
            logger.info(f"\n處理資料集: {dataset_name}")
            logger.info(f"  資料形狀: X={dataset['X'].shape}, y={dataset['y'].shape}")
            logger.info(
                f"  類別分布: 健康={sum(dataset['y']==0)}, 病患={sum(dataset['y']==1)}"
            )

            dataset_results = {}

            # 對每個CV方法
            for cv_method in self.cv_methods:
                logger.info(f"\n  使用 {cv_method} 交叉驗證")

                cv_results = {}

                # 對每個分類器
                for classifier_type in self.classifiers:
                    logger.info(f"    訓練 {classifier_type.value}...")

                    try:
                        result = self._train_single(
                            dataset["X"],
                            dataset["y"],
                            dataset["subject_ids"],
                            cv_method,
                            classifier_type,
                        )
                        cv_results[classifier_type.value] = result

                        logger.info(
                            f"      Acc={result['accuracy']:.3f}, MCC={result['mcc']:.3f}"
                        )

                    except Exception as e:
                        logger.error(f"      失敗: {e}")
                        cv_results[classifier_type.value] = {"error": str(e)}

                dataset_results[cv_method] = cv_results

            all_results[dataset_name] = dataset_results

        return all_results

    def _train_single(
        self, X, y, subject_ids, cv_method: str, classifier_type: ClassifierType
    ) -> Dict:
        """訓練單一配置

        Returns:
            包含評估指標的字典
        """
        # 設定CV
        if "LOSO" in cv_method.upper():
            cv_enum = CVMethod.LOSO
            n_folds = None
        else:
            cv_enum = CVMethod.KFOLD
            n_folds = 5 if "5" in cv_method else 10

        # 設定特徵選擇（根據分類器類型）
        if classifier_type in [ClassifierType.SVM, ClassifierType.LOGISTIC]:
            feat_config = SelectionConfig(
                method=SelectionMethod.CORRELATION, correlation_threshold=0.95
            )
        elif classifier_type == ClassifierType.XGB:
            feat_config = SelectionConfig(
                method=SelectionMethod.XGB_IMPORTANCE, importance_ratio=0.8
            )
        else:
            feat_config = None

        # CV配置
        cv_config = CVConfig(
            method=cv_enum, n_folds=n_folds, feature_selection=feat_config
        )

        # 分類器配置
        clf_config = ClassifierConfig.get_default(classifier_type)

        # 執行交叉驗證
        validator = CrossValidator(cv_config)
        results = validator.validate(X, y, subject_ids, clf_config)

        return {
            "accuracy": results.accuracy,
            "mcc": results.mcc,
            "sensitivity": results.sensitivity,
            "specificity": results.specificity,
            "confusion_matrix": results.confusion_matrix.tolist(),
        }
