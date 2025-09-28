# src/train/trainer.py
"""訓練管線整合器"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass
import json
import numpy as np
from datetime import datetime

from src.data.loader import FeatureDataLoader
from src.data.demographics import DemographicsProcessor
from src.data.balancing import DataBalancer, BalancingConfig
from src.features.selection import SelectionConfig
from src.features.demographic_features import DemographicFeatureIntegrator, DemographicFeatureConfig
from src.models.classifiers import ClassifierConfig
from src.train.cross_validation import CrossValidator, CVConfig, CVMethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """訓練配置"""
    # 資料配置
    embedding_models: List[str] = None
    feature_types: List[str] = None
    use_all_visits: bool = False
    
    # 平衡配置
    enable_age_matching: bool = True
    enable_cdr_filter: bool = False
    cdr_thresholds: List[float] = None
    
    # 人口學特徵
    include_demographics: bool = True
    demo_weight: float = 1.0
    
    # 分類器
    classifier_types: List[str] = None
    
    # 交叉驗證
    cv_method: str = "LOSO"
    n_folds: int = 5
    
    # 特徵選擇
    use_feature_selection: bool = True
    
    # 輸出
    output_dir: str = "results"
    
    def __post_init__(self):
        if self.embedding_models is None:
            self.embedding_models = ['vggface', 'arcface', 'dlib', 'deepid']
        if self.feature_types is None:
            self.feature_types = ['difference', 'average', 'relative']
        if self.classifier_types is None:
            self.classifier_types = ['Random Forest', 'XGBoost', 'SVM', 'Logistic Regression']
        if self.cdr_thresholds is None:
            self.cdr_thresholds = [None] if not self.enable_cdr_filter else [0.5, 1.0]


class Trainer:
    """訓練管線執行器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results = {}
        
    def run(
        self,
        data_path: str,
        demographics_processor: DemographicsProcessor
    ) -> Dict[str, Any]:
        """執行完整訓練流程"""
        logger.info("="*60)
        logger.info("開始訓練流程")
        logger.info("="*60)
        
        # 載入資料
        loader = FeatureDataLoader(data_path)
        
        overall_results = {}
        
        # 對每個CDR閾值
        for cdr_threshold in self.config.cdr_thresholds:
            cdr_results = {}
            
            if self.config.enable_cdr_filter and cdr_threshold is not None:
                logger.info(f"\n使用 CDR > {cdr_threshold} 篩選")
            
            # 資料平衡
            balance_config = BalancingConfig(
                enable_age_matching=self.config.enable_age_matching,
                enable_cdr_filter=self.config.enable_cdr_filter,
                cdr_threshold=cdr_threshold
            )
            balancer = DataBalancer(demographics_processor, balance_config)
            allowed_ids, age_summary = balancer.balance_groups()
            
            # 建立人口學查詢表
            demo_lookup = demographics_processor.build_lookup_table()
            
            # 對每個嵌入模型和特徵類型
            for embedding_model in self.config.embedding_models:
                for feature_type in self.config.feature_types:
                    model_key = f"{embedding_model}_{feature_type}"
                    logger.info(f"\n處理: {model_key}")
                    
                    # 載入特徵
                    subjects = loader.scan_subjects(use_all_visits=self.config.use_all_visits)
                    feature_data = loader.load_features(
                        subjects,
                        embedding_model=embedding_model,
                        feature_type=feature_type
                    )
                    
                    # 準備資料
                    X, y, subject_ids = self._prepare_data(
                        feature_data,
                        allowed_ids,
                        demo_lookup
                    )
                    
                    if len(X) == 0:
                        logger.warning(f"  {model_key} 沒有資料，跳過")
                        continue
                    
                    logger.info(f"  資料: {X.shape}, 健康={np.sum(y==0)}, 病患={np.sum(y==1)}")
                    
                    # 訓練不同分類器
                    model_results = self._train_classifiers(X, y, subject_ids)
                    cdr_results[model_key] = model_results
            
            # 儲存該CDR閾值的結果
            result_key = f"CDR_gt_{cdr_threshold}" if cdr_threshold else "No_filtering"
            overall_results[result_key] = cdr_results
        
        # 儲存結果
        self._save_results(overall_results)
        
        return overall_results
    
    def _prepare_data(
        self,
        feature_data: List,
        allowed_ids: Dict[str, Set[str]],
        demo_lookup: Dict
    ) -> tuple:
        """準備訓練資料"""
        X_list = []
        y_list = []
        subject_ids = []
        
        for fd in feature_data:
            # 檢查是否在允許清單中
            if allowed_ids:
                subject_id = fd.subject_info.subject_id
                group = fd.subject_info.group
                if group in allowed_ids and subject_id not in allowed_ids[group]:
                    # 也檢查base_id
                    from src.utils.id_parser import parse_subject_id
                    base_id, _ = parse_subject_id(subject_id)
                    if base_id not in allowed_ids[group]:
                        continue
            
            features = list(fd.features.values())[0]
            X_list.append(features)
            y_list.append(fd.subject_info.label)
            subject_ids.append(fd.subject_info.subject_id)
        
        if not X_list:
            return np.array([]), np.array([]), []
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 整合人口學特徵
        if self.config.include_demographics:
            demo_config = DemographicFeatureConfig(
                weight=self.config.demo_weight
            )
            integrator = DemographicFeatureIntegrator(demo_config)
            X = integrator.integrate_features(X, subject_ids, demo_lookup)
        
        return X, y, subject_ids
    
    def _train_classifiers(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str]
    ) -> Dict:
        """訓練所有分類器"""
        results = {}
        
        for classifier_name in self.config.classifier_types:
            logger.info(f"    訓練 {classifier_name}...")
            
            # 設定交叉驗證
            cv_method = CVMethod.LOSO if self.config.cv_method == "LOSO" else CVMethod.KFOLD
            cv_config = CVConfig(
                method=cv_method,
                n_folds=self.config.n_folds
            )
            
            # 設定特徵選擇
            if self.config.use_feature_selection:
                from src.features.selection import FeatureSelector
                method = FeatureSelector.get_method_for_classifier(classifier_name)
                cv_config.feature_selection = SelectionConfig(method=method)
            
            # 執行交叉驗證
            validator = CrossValidator(cv_config)
            from src.models.classifiers import ClassifierFactory
            clf_type = ClassifierFactory.from_string(classifier_name)
            clf_config = ClassifierConfig.get_default(clf_type)
            
            cv_results = validator.validate(X, y, subject_ids, clf_config)
            
            # 儲存結果
            results[classifier_name] = cv_results.to_dict()
            
            # 顯示結果
            logger.info(f"      準確率: {cv_results.accuracy:.4f}")
            logger.info(f"      MCC: {cv_results.mcc:.4f}")
        
        return results
    
    def _save_results(self, results: Dict):
        """儲存結果"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 建立檔名
        suffix_parts = []
        if not self.config.enable_age_matching:
            suffix_parts.append("no_age")
        if self.config.enable_cdr_filter:
            suffix_parts.append("cdr")
        if self.config.use_all_visits:
            suffix_parts.append("all_visits")
        suffix_parts.append(self.config.cv_method.lower())
        
        suffix = "_".join(suffix_parts) if suffix_parts else "default"
        filename = output_path / f"results_{suffix}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n結果已儲存至: {filename}")