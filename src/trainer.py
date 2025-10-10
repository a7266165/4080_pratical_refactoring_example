# src/modeltrainer.py
"""整合的模型訓練模組"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from datetime import datetime
from enum import Enum
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from src.utils import calculate_metrics, save_json

logger = logging.getLogger(__name__)


# ========== Training Utils 部分 ==========
# ==================== 分類器定義 ====================
class ClassifierType(Enum):
    """支援的分類器類型"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    SVM = "svm"
    LOGISTIC = "logistic"

# 分類器配置
CLASSIFIER_CONFIG = {
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        "needs_scaling": False
    },
    "xgboost": {
        "class": xgb.XGBClassifier,
        "params": {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        },
        "needs_scaling": False
    },
    "svm": {
        "class": SVC,
        "params": {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42,
            'probability': True
        },
        "needs_scaling": True
    },
    "logistic": {
        "class": LogisticRegression,
        "params": {
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        },
        "needs_scaling": True
    }
}

# ==================== 分類器工廠 ====================
def create_classifier(classifier_name: str, custom_params: Optional[Dict] = None):
    """創建分類器實例
    
    Args:
        classifier_name: 分類器名稱
        custom_params: 自定義參數（可選）
    
    Returns:
        分類器實例
    """
    if classifier_name not in CLASSIFIER_CONFIG:
        raise ValueError(f"未知的分類器: {classifier_name}")
    
    config = CLASSIFIER_CONFIG[classifier_name]
    params = custom_params or config['params']
    
    logger.debug(f"創建分類器: {classifier_name}")
    return config['class'](**params)

def prepare_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    classifier_name: str
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """準備訓練和測試資料（根據需要進行標準化）
    
    Args:
        X_train: 訓練特徵
        X_test: 測試特徵
        classifier_name: 分類器名稱
    
    Returns:
        處理後的訓練資料、測試資料、標準化器（如果使用）
    """
    if CLASSIFIER_CONFIG[classifier_name]['needs_scaling']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.debug(f"{classifier_name} 使用標準化")
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train, X_test, None

# ==================== 特徵選擇 ====================
def select_features_by_correlation(
    X: np.ndarray,
    threshold: float = 0.95
) -> Tuple[np.ndarray, List[int]]:
    """基於相關性移除高度相關的特徵
    
    Args:
        X: 特徵矩陣
        threshold: 相關係數閾值
    
    Returns:
        篩選後的特徵矩陣和保留的特徵索引
    """
    n_features = X.shape[1]
    if n_features <= 1:
        return X, list(range(n_features))
    
    # 計算相關矩陣
    corr_matrix = np.corrcoef(X.T)
    
    # 找出要保留的特徵
    keep_features = []
    removed_features: Set[int] = set()
    
    for i in range(corr_matrix.shape[0]):
        if i in removed_features:
            continue
        keep_features.append(i)
        
        # 移除高度相關的特徵
        for j in range(i + 1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > threshold:
                removed_features.add(j)
    
    logger.info(f"相關性過濾: {n_features} -> {len(keep_features)} 個特徵")
    
    return X[:, keep_features], keep_features

def select_features_by_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    importance_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """基於XGBoost特徵重要性選擇特徵
    
    Args:
        X_train: 訓練特徵
        y_train: 訓練標籤
        X_test: 測試特徵
        importance_ratio: 保留的累積重要性比例
    
    Returns:
        篩選後的訓練特徵、測試特徵、選擇的特徵索引
    """
    # 訓練臨時XGBoost模型獲取重要性
    temp_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    temp_model.fit(X_train, y_train)
    
    # 獲取特徵重要性
    importance = temp_model.feature_importances_
    
    # 排序並計算累積重要性
    indices = np.argsort(importance)[::-1]
    cumsum = np.cumsum(importance[indices])
    
    # 找出達到閾值的特徵數量
    if cumsum[-1] > 0:
        n_features = np.searchsorted(cumsum, importance_ratio * cumsum[-1]) + 1
    else:
        n_features = len(indices)
    
    n_features = max(1, min(n_features, len(indices)))
    
    # 選擇最重要的特徵
    selected_features = sorted(indices[:n_features])
    
    logger.info(
        f"XGBoost特徵選擇: {X_train.shape[1]} -> {len(selected_features)} 個特徵"
    )
    
    return (
        X_train[:, selected_features],
        X_test[:, selected_features],
        selected_features
    )

# ==================== 交叉驗證 ====================
class CrossValidator:
    """交叉驗證器"""
    
    def __init__(
        self,
        cv_method: str = "5-fold",
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            cv_method: 交叉驗證方法 ("5-fold" 或 "loso")
            n_folds: 折數
            random_state: 隨機種子
        """
        self.cv_method = cv_method.lower()
        self.n_folds = n_folds
        self.random_state = random_state
    
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_name: str,
        use_feature_selection: bool = False
    ) -> Dict:
        """執行交叉驗證
        
        Args:
            X: 特徵矩陣
            y: 標籤
            subject_ids: 受試者ID列表
            classifier_name: 分類器名稱
            use_feature_selection: 是否使用特徵選擇
        
        Returns:
            評估指標字典
        """
        if "loso" in self.cv_method:
            return self._loso_cv(
                X, y, subject_ids, classifier_name, use_feature_selection
            )
        else:
            return self._kfold_cv(
                X, y, subject_ids, classifier_name, use_feature_selection
            )
    
    def _loso_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_name: str,
        use_feature_selection: bool
    ) -> Dict:
        """Leave-One-Subject-Out 交叉驗證"""
        unique_subjects = list(set(subject_ids))
        n_subjects = len(unique_subjects)
        
        logger.info(f"執行 LOSO 交叉驗證，共 {n_subjects} 個受試者")
        
        all_y_true = []
        all_y_pred = []
        
        for i, test_subject in enumerate(unique_subjects):
            # 分割資料
            test_idx = [j for j, sid in enumerate(subject_ids) if sid == test_subject]
            train_idx = [j for j, sid in enumerate(subject_ids) if sid != test_subject]
            
            if not test_idx or not train_idx:
                continue
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 特徵選擇
            if use_feature_selection and classifier_name == "xgboost":
                X_train, X_test, _ = select_features_by_importance(
                    X_train, y_train, X_test
                )
            elif use_feature_selection:
                selected_features = select_features_by_correlation(X_train)[1]
                X_train = X_train[:, selected_features]
                X_test = X_test[:, selected_features]
            
            # 資料預處理
            X_train, X_test, _ = prepare_data(X_train, X_test, classifier_name)
            
            # 訓練和預測
            classifier = create_classifier(classifier_name)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            
            # 多數決（如果有多個樣本）
            y_pred_final = 1 if np.mean(y_pred) >= 0.5 else 0
            y_true_final = 1 if np.mean(y_test) >= 0.5 else 0
            
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
        use_feature_selection: bool
    ) -> Dict:
        """K-Fold 交叉驗證（按受試者分組）"""
        logger.info(f"執行 {self.n_folds}-fold 交叉驗證")
        
        # 建立受試者級別的資料
        unique_subjects = list(set(subject_ids))
        subject_to_indices = {subj: [] for subj in unique_subjects}
        for idx, subj in enumerate(subject_ids):
            subject_to_indices[subj].append(idx)
        
        # 受試者級別的標籤
        subject_labels = []
        for subj in unique_subjects:
            indices = subject_to_indices[subj]
            labels = [y[i] for i in indices]
            subject_labels.append(1 if np.mean(labels) >= 0.5 else 0)
        
        subject_labels = np.array(subject_labels)
        
        # 分層K折
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        all_y_true = []
        all_y_pred = []
        
        for fold_idx, (train_subj_idx, test_subj_idx) in enumerate(
            skf.split(unique_subjects, subject_labels)
        ):
            # 獲取訓練和測試索引
            train_idx = []
            test_idx = []
            test_subjects = [unique_subjects[i] for i in test_subj_idx]
            
            for i in train_subj_idx:
                train_idx.extend(subject_to_indices[unique_subjects[i]])
            for i in test_subj_idx:
                test_idx.extend(subject_to_indices[unique_subjects[i]])
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 特徵選擇
            if use_feature_selection and classifier_name == "xgboost":
                X_train, X_test, _ = select_features_by_importance(
                    X_train, y_train, X_test
                )
            elif use_feature_selection:
                selected_features = select_features_by_correlation(X_train)[1]
                X_train = X_train[:, selected_features]
                X_test = X_test[:, selected_features]
            
            # 資料預處理
            X_train, X_test, _ = prepare_data(X_train, X_test, classifier_name)
            
            # 訓練分類器
            classifier = create_classifier(classifier_name)
            classifier.fit(X_train, y_train)
            
            # 對每個測試受試者預測（多數決）
            for test_subj in test_subjects:
                subj_test_idx = [
                    test_idx.index(i)
                    for i in test_idx
                    if subject_ids[i] == test_subj
                ]
                
                y_pred_subj = classifier.predict(X_test[subj_test_idx])
                y_true_subj = y_test[subj_test_idx]
                
                y_pred_final = 1 if np.mean(y_pred_subj) >= 0.5 else 0
                y_true_final = 1 if np.mean(y_true_subj) >= 0.5 else 0
                
                all_y_true.append(y_true_final)
                all_y_pred.append(y_pred_final)
        
        return calculate_metrics(np.array(all_y_true), np.array(all_y_pred))

# ========== ModelTrainer 部分 ==========
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
        
        # ⭐ 新增：XGBoost特徵選擇（針對vggface的4096維）
        selected_features = None
        if model_type == "xgboost" and X_train.shape[1] > 1000:  # 高維度才做選擇
            logger.info("執行XGBoost特徵選擇...")
            X_train, X_test, selected_features = select_features_by_importance(
                X_train, y_train, X_test,
                importance_ratio=0.8  # 保留80%重要性
            )
            logger.info(f"特徵選擇後: 訓練集 {X_train.shape}, 測試集 {X_test.shape}")

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
            "selected_features": selected_features,
            "X_train_shape": tuple(int(x) for x in X_train.shape),  # 轉換為 tuple of int
            "X_test_shape": tuple(int(x) for x in X_test.shape),    # 轉換為 tuple of int
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_results(self, config_key: str, results: Dict):
        """儲存模型和訓練結果"""
        
        # 1. 儲存模型
        model = results["model"]
        model_type = results["model_type"]
        
        if model_type == "xgboost":
            # XGBoost使用JSON格式
            model_path = self.models_dir / f"{config_key}.json"
            model.save_model(str(model_path))
            logger.info(f"XGBoost模型已儲存(JSON): {model_path}")
        else:
            # 其他模型使用pickle格式
            model_path = self.models_dir / f"{config_key}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"模型已儲存(PKL): {model_path}")
    
        
        # ⭐ 新增：儲存特徵選擇資訊（給API用）
        if results.get("selected_features") is not None:
            feature_selection_info = {
                "selected_indices": results["selected_features"],
                "original_dim": 4096,  # VGGFace維度
                "selected_dim": len(results["selected_features"]),
                "importance_ratio": 0.8
            }
            feature_path = self.models_dir / f"{config_key}_features.json"
            save_json(feature_selection_info, feature_path)
            logger.info(f"特徵選擇資訊已儲存: {feature_path}")
        
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

            # 特徵選擇資訊
            if results.get("selected_features") is not None:
                f.write(f"\n特徵選擇:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  原始維度: 4096\n")
                f.write(f"  選擇維度: {len(results['selected_features'])}\n")
                f.write(f"  壓縮比例: {len(results['selected_features'])/4096:.1%}\n")

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