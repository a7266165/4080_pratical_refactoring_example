# src/modeltrainer/modeltrainer.py
"""模型訓練與儲存模組"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from src.utils.utils import calculate_metrics, save_json

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型訓練器

    負責：
    1. 訓練多種分類模型
    2. 評估模型效能
    3. 儲存模型與相關檔案
    4. 記錄訓練歷程
    """

    def __init__(
        self,
        output_dir: str,
        model_types: Optional[Union[str, List[str]]] = None,
        random_state: int = 42,
    ):
        """
        初始化訓練器

        Args:
            output_dir: 輸出目錄路徑
            model_types: 模型類型（字串或列表），支援 "random_forest", "svm", "logistic", "xgboost"
            random_state: 隨機種子
        """
        self.output_dir = Path(output_dir)

        # 處理模型類型（支援單一字串或列表）
        if model_types is None:
            self.model_types = ["random_forest"]
        elif isinstance(model_types, str):
            self.model_types = [model_types]
        else:
            self.model_types = model_types

        self.random_state = random_state

        # 建立輸出目錄結構
        self._setup_output_dirs()

        logger.info(f"初始化 ModelTrainer，模型類型: {self.model_types}")

    def _setup_output_dirs(self):
        """建立輸出目錄結構"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 建立子目錄
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.configs_dir = self.output_dir / "configs"

        for dir_path in [self.models_dir, self.reports_dir, self.configs_dir]:
            dir_path.mkdir(exist_ok=True)

        logger.info(f"輸出目錄設置完成: {self.output_dir}")

    def _initialize_model(self, model_type: str):
        """初始化分類模型

        Args:
            model_type: 模型類型名稱
        """
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif model_type == "svm":
            return SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=self.random_state,
                probability=True,  # 需要預測機率
            )
        elif model_type == "logistic":
            return LogisticRegression(
                max_iter=1000, random_state=self.random_state, n_jobs=-1
            )
        elif model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:
            raise ValueError(f"不支援的模型類型: {model_type}")

    def train_and_save(
        self, datasets: Dict[str, Dict], test_size: float = 0.2, cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        訓練並儲存所有資料集配置和模型類型的組合

        Args:
            datasets: 資料集字典（從DataLoader.load()取得）
            test_size: 測試集比例
            cv_folds: 交叉驗證折數

        Returns:
            所有模型的訓練結果
        """
        all_results = {}
        total_configs = len(datasets) * len(self.model_types)
        current_config = 0

        logger.info(
            f"開始訓練 {total_configs} 個配置 "
            f"({len(datasets)} 個資料集 × {len(self.model_types)} 個模型類型)"
        )

        for dataset_key, dataset in datasets.items():
            for model_type in self.model_types:
                current_config += 1

                # 建立完整的配置key
                full_key = f"{dataset_key}_{model_type}"

                logger.info(
                    f"\n[{current_config}/{total_configs}] 訓練模型: {full_key}"
                )
                logger.info("=" * 50)

                try:
                    # 訓練單一配置
                    results = self._train_single_dataset(
                        dataset_key=dataset_key,
                        model_type=model_type,
                        X=dataset["X"],
                        y=dataset["y"],
                        subject_ids=dataset.get("subject_ids", []),
                        metadata=dataset.get("metadata", {}),
                        test_size=test_size,
                        cv_folds=cv_folds,
                    )

                    all_results[full_key] = results

                    # 儲存模型與結果
                    self._save_model_and_results(full_key, results)

                except Exception as e:
                    logger.error(f"訓練 {full_key} 時發生錯誤: {str(e)}")
                    continue

        # 儲存總結報告
        self._save_summary_report(all_results)

        return all_results

    def _train_single_dataset(
        self,
        dataset_key: str,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: list,
        metadata: dict,
        test_size: float,
        cv_folds: int,
    ) -> Dict[str, Any]:
        """訓練單一資料集配置與模型類型組合"""

        # 分割訓練集與測試集
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            range(len(y)),
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # 取得對應的subject_ids
        train_ids = [subject_ids[i] for i in idx_train] if subject_ids else []
        test_ids = [subject_ids[i] for i in idx_test] if subject_ids else []

        # 訓練模型
        logger.info(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")

        # 初始化指定類型的模型
        model = self._initialize_model(model_type)
        model.fit(X_train, y_train)

        # 預測
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # 預測機率（如果模型支援）
        try:
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_prob_test = model.predict_proba(X_test)[:, 1]
        except:
            y_prob_train = None
            y_prob_test = None

        # 計算評估指標
        train_metrics = calculate_metrics(y_train, y_pred_train, y_prob_train)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test)

        # 交叉驗證
        cv_scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring="accuracy", n_jobs=-1
        )

        logger.info(f"訓練準確率: {train_metrics['accuracy']:.3f}")
        logger.info(f"測試準確率: {test_metrics['accuracy']:.3f}")
        logger.info(f"測試 MCC: {test_metrics['mcc']:.3f}")
        logger.info(f"測試 F1: {test_metrics['f1']:.3f}")
        logger.info(
            f"交叉驗證準確率: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})"
        )

        # 特徵重要性（如果模型支援）
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_

        return {
            "model": model,
            "dataset_key": dataset_key,
            "model_type": model_type,
            "metadata": metadata,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "feature_importance": feature_importance,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "timestamp": datetime.now().isoformat(),
        }

    def _save_model_and_results(self, full_key: str, results: Dict[str, Any]):
        """儲存模型與訓練結果"""

        # 1. 儲存模型檔案
        model_filename = f"{full_key}.pkl"
        model_path = self.models_dir / model_filename

        with open(model_path, "wb") as f:
            pickle.dump(results["model"], f)
        logger.info(f"模型已儲存: {model_path}")

        # 2. 儲存訓練結果（不含模型物件）
        results_for_save = {k: v for k, v in results.items() if k != "model"}

        # 轉換numpy arrays為list以便JSON序列化
        if results_for_save.get("feature_importance") is not None:
            results_for_save["feature_importance"] = results_for_save[
                "feature_importance"
            ].tolist()

        results_filename = f"{full_key}_results.json"
        results_path = self.reports_dir / results_filename

        save_json(results_for_save, results_path)
        logger.info(f"訓練結果已儲存: {results_path}")

        # 3. 儲存詳細報告
        report_filename = f"{full_key}_report.txt"
        report_path = self.reports_dir / report_filename

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"模型訓練報告\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"資料集: {results['dataset_key']}\n")
            f.write(f"模型類型: {results['model_type']}\n")
            f.write(f"訓練時間: {results['timestamp']}\n")
            f.write(f"訓練集大小: {results['X_train_shape']}\n")
            f.write(f"測試集大小: {results['X_test_shape']}\n\n")

            f.write(f"訓練集效能:\n")
            f.write(f"-" * 30 + "\n")
            for metric, value in results["train_metrics"].items():
                if metric != "confusion_matrix":
                    f.write(f"  {metric}: {value:.4f}\n")

            f.write(f"\n測試集效能:\n")
            f.write(f"-" * 30 + "\n")
            for metric, value in results["test_metrics"].items():
                if metric != "confusion_matrix":
                    f.write(f"  {metric}: {value:.4f}\n")

            f.write(f"\n交叉驗證:\n")
            f.write(f"-" * 30 + "\n")
            f.write(
                f"  平均準確率: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})\n"
            )

            # 特別強調 MCC
            f.write(f"\n關鍵指標:\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"  測試 MCC: {results['test_metrics']['mcc']:.4f}\n")
            f.write(f"  測試 F1: {results['test_metrics']['f1']:.4f}\n")
            if "auc" in results["test_metrics"]:
                f.write(f"  測試 AUC: {results['test_metrics']['auc']:.4f}\n")

            # 混淆矩陣
            f.write(f"\n測試集混淆矩陣:\n")
            f.write(f"-" * 30 + "\n")
            cm = results["test_metrics"]["confusion_matrix"]
            f.write(f"  預測負類  預測正類\n")
            f.write(f"實際負類   {cm[0][0]:4d}     {cm[0][1]:4d}\n")
            f.write(f"實際正類   {cm[1][0]:4d}     {cm[1][1]:4d}\n")

        logger.info(f"詳細報告已儲存: {report_path}")

    def _save_summary_report(self, all_results: Dict[str, Any]):
        """儲存總結報告"""

        summary_path = self.output_dir / "training_summary.json"
        summary = {
            "model_types": self.model_types,
            "timestamp": datetime.now().isoformat(),
            "num_configurations": len(all_results),
            "results": {},
        }

        # 整理每個資料集的關鍵指標
        for config_key, results in all_results.items():
            summary["results"][config_key] = {
                "train_accuracy": results["train_metrics"]["accuracy"],
                "test_accuracy": results["test_metrics"]["accuracy"],
                "test_mcc": results["test_metrics"]["mcc"],  # 加入 MCC
                "test_f1": results["test_metrics"]["f1"],
                "test_precision": results["test_metrics"]["precision"],
                "test_recall": results["test_metrics"]["recall"],
                "cv_mean": results["cv_mean"],
                "cv_std": results["cv_std"],
                "test_auc": results["test_metrics"].get("auc", None),
            }

        save_json(summary, summary_path)

        logger.info(f"總結報告已儲存: {summary_path}")
