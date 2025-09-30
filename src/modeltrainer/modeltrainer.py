# src/modeltrainer/modeltrainer.py
"""簡化版模型訓練器"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

from src.modeltrainer.trainingutils import (
    create_classifier,
    prepare_data
)
from src.utils.utils import calculate_metrics, save_json

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型訓練器 - 簡化版"""
    
    def __init__(
        self,
        output_dir: str,
        model_types: Optional[Union[str, List[str]]] = None,
        random_state: int = 42
    ):
        """
        Args:
            output_dir: 輸出目錄
            model_types: 模型類型（字串或列表）
            random_state: 隨機種子
        """
        self.output_dir = Path(output_dir)
        
        # 處理模型類型
        if model_types is None:
            self.model_types = ["random_forest"]
        elif isinstance(model_types, str):
            self.model_types = [model_types]
        else:
            self.model_types = model_types
        
        self.random_state = random_state
        
        # 建立輸出目錄
        self._setup_dirs()
        
        logger.info(f"初始化訓練器，模型類型: {self.model_types}")
    
    def _setup_dirs(self):
        """建立輸出目錄結構"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    def train_and_save(
        self,
        datasets: Dict[str, Dict],
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """訓練所有配置的模型
        
        Args:
            datasets: 資料集字典
            test_size: 測試集比例
            cv_folds: 交叉驗證折數
        
        Returns:
            所有模型的訓練結果
        """
        all_results = {}
        total_configs = len(datasets) * len(self.model_types)
        current = 0
        
        logger.info(
            f"開始訓練 {total_configs} 個配置 "
            f"({len(datasets)} 資料集 × {len(self.model_types)} 模型)"
        )
        
        for dataset_key, dataset in datasets.items():
            for model_type in self.model_types:
                current += 1
                config_key = f"{dataset_key}_{model_type}"
                
                logger.info(f"\n[{current}/{total_configs}] 訓練: {config_key}")
                logger.info("=" * 50)
                
                try:
                    # 訓練模型
                    results = self._train_single(
                        dataset,
                        model_type,
                        test_size,
                        cv_folds
                    )
                    
                    all_results[config_key] = results
                    
                    # 儲存模型和結果
                    self._save_results(config_key, results)
                    
                except Exception as e:
                    logger.error(f"訓練 {config_key} 失敗: {str(e)}")
                    continue
        
        # 儲存總結報告
        self._save_summary(all_results)
        
        return all_results
    
    def _train_single(
        self,
        dataset: Dict,
        model_type: str,
        test_size: float,
        cv_folds: int
    ) -> Dict[str, Any]:
        """訓練單一模型配置"""
        
        X = dataset["X"]
        y = dataset["y"]
        subject_ids = dataset.get("subject_ids", [])
        metadata = dataset.get("metadata", {})
        
        # 分割資料集
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, range(len(y)),
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")
        
        # 資料預處理
        X_train, X_test, scaler = prepare_data(X_train, X_test, model_type)
        
        # 訓練模型
        model = create_classifier(model_type)
        model.fit(X_train, y_train)
        
        # 預測
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 預測機率（如果支援）
        try:
            y_prob_test = model.predict_proba(X_test)[:, 1]
        except:
            y_prob_test = None
        
        # 計算評估指標
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test)
        
        # 交叉驗證
        cv_scores = cross_val_score(
            model, X, y, 
            cv=cv_folds, 
            scoring="accuracy",
            n_jobs=-1
        )
        
        # 記錄結果
        logger.info(f"訓練準確率: {train_metrics['accuracy']:.3f}")
        logger.info(f"測試準確率: {test_metrics['accuracy']:.3f}")
        logger.info(f"測試 MCC: {test_metrics['mcc']:.3f}")
        logger.info(f"測試 F1: {test_metrics['f1']:.3f}")
        logger.info(
            f"交叉驗證: {cv_scores.mean():.3f} "
            f"(+/- {cv_scores.std():.3f})"
        )
        
        # 特徵重要性（如果支援）
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_.tolist()
        
        return {
            "model": model,
            "model_type": model_type,
            "metadata": metadata,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),  # 轉換為 float
            "cv_std": float(cv_scores.std()),     # 轉換為 float
            "feature_importance": feature_importance,
            "X_train_shape": tuple(int(x) for x in X_train.shape),  # 轉換為 tuple of int
            "X_test_shape": tuple(int(x) for x in X_test.shape),    # 轉換為 tuple of int
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_results(self, config_key: str, results: Dict):
        """儲存模型和訓練結果"""
        
        # 1. 儲存模型
        model_path = self.models_dir / f"{config_key}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(results["model"], f)
        logger.info(f"模型已儲存: {model_path}")
        
        # 2. 儲存結果JSON（不含模型物件）
        results_for_save = {
            k: v for k, v in results.items() 
            if k != "model"
        }
        
        results_path = self.reports_dir / f"{config_key}_results.json"
        save_json(results_for_save, results_path)
        
        # 3. 儲存文字報告
        self._save_text_report(config_key, results)
    
    def _save_text_report(self, config_key: str, results: Dict):
        """儲存文字格式報告"""
        report_path = self.reports_dir / f"{config_key}_report.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("模型訓練報告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"配置: {config_key}\n")
            f.write(f"模型類型: {results['model_type']}\n")
            f.write(f"訓練時間: {results['timestamp']}\n")
            f.write(f"訓練集大小: {results['X_train_shape']}\n")
            f.write(f"測試集大小: {results['X_test_shape']}\n\n")
            
            # 訓練集效能
            f.write("訓練集效能:\n")
            f.write("-" * 30 + "\n")
            for metric, value in results["train_metrics"].items():
                if metric != "confusion_matrix":
                    f.write(f"  {metric}: {value:.4f}\n")
            
            # 測試集效能
            f.write("\n測試集效能:\n")
            f.write("-" * 30 + "\n")
            for metric, value in results["test_metrics"].items():
                if metric != "confusion_matrix":
                    f.write(f"  {metric}: {value:.4f}\n")
            
            # 交叉驗證
            f.write("\n交叉驗證:\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"  平均準確率: {results['cv_mean']:.4f} "
                f"(+/- {results['cv_std']:.4f})\n"
            )
            
            # 關鍵指標
            f.write("\n關鍵指標:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  測試 MCC: {results['test_metrics']['mcc']:.4f}\n")
            f.write(f"  測試 F1: {results['test_metrics']['f1']:.4f}\n")
            f.write(
                f"  敏感度: {results['test_metrics']['sensitivity']:.4f}\n"
            )
            f.write(
                f"  特異度: {results['test_metrics']['specificity']:.4f}\n"
            )
            
            # 顯示文字報告
            f.write("\n測試集混淆矩陣:\n")
            f.write("-" * 30 + "\n")
            cm = results["test_metrics"]["confusion_matrix"]
            f.write("         預測負  預測正\n")
            f.write(f"實際負   {int(cm[0][0]):5d}   {int(cm[0][1]):5d}\n")
            f.write(f"實際正   {int(cm[1][0]):5d}   {int(cm[1][1]):5d}\n")
    
    def _save_summary(self, all_results: Dict):
        """儲存總結報告"""
        summary_path = self.output_dir / "training_summary.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_types": self.model_types,
            "num_configurations": len(all_results),
            "results": {}
        }
        
        # 整理每個配置的關鍵指標
        for config_key, results in all_results.items():
            summary["results"][config_key] = {
                "train_accuracy": results["train_metrics"]["accuracy"],
                "test_accuracy": results["test_metrics"]["accuracy"],
                "test_mcc": results["test_metrics"]["mcc"],
                "test_f1": results["test_metrics"]["f1"],
                "test_sensitivity": results["test_metrics"]["sensitivity"],
                "test_specificity": results["test_metrics"]["specificity"],
                "cv_mean": results["cv_mean"],
                "cv_std": results["cv_std"]
            }
        
        save_json(summary, summary_path)
        logger.info(f"\n總結報告已儲存: {summary_path}")
        
        # 找出最佳模型
        self._print_best_results(summary["results"])
    
    def _print_best_results(self, results_summary: Dict):
        """顯示最佳模型結果"""
        
        if not results_summary:
            return
        
        # 找出最佳MCC
        best_mcc_key = max(
            results_summary.keys(),
            key=lambda k: results_summary[k]["test_mcc"]
        )
        
        # 找出最佳準確率
        best_acc_key = max(
            results_summary.keys(),
            key=lambda k: results_summary[k]["test_accuracy"]
        )
        
        print("\n" + "=" * 60)
        print("最佳模型結果")
        print("=" * 60)
        
        # 顯示最佳MCC模型
        best_mcc = results_summary[best_mcc_key]
        print(f"\n最佳MCC: {best_mcc_key}")
        print(f"  MCC: {best_mcc['test_mcc']:.4f}")
        print(f"  準確率: {best_mcc['test_accuracy']:.4f}")
        print(f"  F1: {best_mcc['test_f1']:.4f}")
        
        # 如果最佳準確率模型不同，也顯示
        if best_acc_key != best_mcc_key:
            best_acc = results_summary[best_acc_key]
            print(f"\n最佳準確率: {best_acc_key}")
            print(f"  準確率: {best_acc['test_accuracy']:.4f}")
            print(f"  MCC: {best_acc['test_mcc']:.4f}")
        
        print("=" * 60)