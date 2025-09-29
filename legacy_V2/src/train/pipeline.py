# src/train/pipeline.py
"""簡化版訓練管線"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import numpy as np
from datetime import datetime
import logging

from legacy_V2.src.data.loader import FeatureDataLoader
from legacy_V2.src.data.demographics import DemographicsProcessor
from legacy_V2.src.data.balancing import DataBalancer, BalancingConfig
from legacy_V2.src.models.classifiers import ClassifierFactory, ClassifierConfig, ClassifierType
from legacy_V2.src.train.cross_validation import CrossValidator, CVConfig, CVMethod
from legacy_V2.src.features.selection import FeatureSelector, SelectionConfig, SelectionMethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimpleTrainingConfig:
    """簡化的訓練配置"""
    # 資料
    embedding_models: List[str] = None
    feature_types: List[str] = None
    use_all_visits: bool = False
    
    # 平衡
    enable_age_matching: bool = False
    enable_cdr_filter: bool = False
    cdr_thresholds: List[float] = None
    
    # 交叉驗證
    cv_method: str = "5-fold"
    
    # 輸出
    output_dir: str = "results"
    
    def __post_init__(self):
        if self.embedding_models is None:
            self.embedding_models = ['vggface']
        if self.feature_types is None:
            self.feature_types = ['difference']
        if self.cdr_thresholds is None:
            self.cdr_thresholds = [None]


class SimplePipeline:
    """簡化的訓練管線"""
    
    def __init__(self, config: SimpleTrainingConfig):
        self.config = config
        self.results = {}
    
    def prepare_data(
        self,
        data_path: str,
        demo_processor: Optional[DemographicsProcessor] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """準備訓練資料"""
        
        # 載入特徵
        loader = FeatureDataLoader(data_path)
        subjects = loader.scan_subjects(use_all_visits=self.config.use_all_visits)
        
        logger.info(f"找到 {len(subjects)} 個樣本")
        
        # 載入第一個配置的特徵
        embedding_model = self.config.embedding_models[0]
        feature_type = self.config.feature_types[0]
        
        feature_data = loader.load_features(
            subjects,
            embedding_model=embedding_model,
            feature_type=feature_type
        )
        
        # 準備特徵矩陣
        X_list = []
        y_list = []
        subject_ids = []
        
        for fd in feature_data:
            features = list(fd.features.values())[0]
            X_list.append(features)
            y_list.append(fd.subject_info.label)
            subject_ids.append(fd.subject_info.subject_id)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 整合人口學特徵（如果有）
        if demo_processor is not None:
            lookup = demo_processor.build_lookup_table()
            X = self._add_demographic_features(X, subject_ids, lookup)
        else:
            lookup = None
        
        logger.info(f"資料準備完成: X={X.shape}, 健康={np.sum(y==0)}, 病患={np.sum(y==1)}")
        
        return X, y, subject_ids, lookup
    
    def _add_demographic_features(
        self,
        X: np.ndarray,
        subject_ids: List[str],
        lookup: Dict
    ) -> np.ndarray:
        """添加人口學特徵"""
        from src.utils.id_parser import parse_subject_id
        
        demo_features = []
        
        for sid in subject_ids:
            meta = lookup.get(sid)
            if meta is None:
                base_id, _ = parse_subject_id(sid)
                meta = lookup.get(base_id)
            
            if meta:
                age = meta.get('Age', np.nan)
                sex = meta.get('Sex', np.nan)
            else:
                age = np.nan
                sex = np.nan
            
            demo_features.append([age, sex])
        
        demo_array = np.array(demo_features)
        
        # 填補缺失值
        age_mean = np.nanmean(demo_array[:, 0])
        sex_mode = np.nanmean(demo_array[:, 1])
        
        if np.isnan(age_mean):
            age_mean = 70
        if np.isnan(sex_mode):
            sex_mode = 0.5
        
        demo_array[np.isnan(demo_array[:, 0]), 0] = age_mean
        demo_array[np.isnan(demo_array[:, 1]), 1] = sex_mode
        
        # 標準化並結合
        from sklearn.preprocessing import StandardScaler
        
        scaler_X = StandardScaler()
        scaler_demo = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        demo_scaled = scaler_demo.fit_transform(demo_array)
        
        X_combined = np.hstack([X_scaled, demo_scaled])
        
        logger.info(f"添加人口學特徵: {X.shape} -> {X_combined.shape}")
        
        return X_combined
    
    def train_all_classifiers(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str]
    ) -> Dict[str, Any]:
        """訓練所有分類器"""
        
        results = {}
        
        # 分類器列表
        classifiers = [
            ClassifierType.RANDOM_FOREST,
            ClassifierType.XGB,
            ClassifierType.SVM,
            ClassifierType.LOGISTIC
        ]
        
        # CV配置
        cv_method = CVMethod.KFOLD if "fold" in self.config.cv_method.lower() else CVMethod.LOSO
        n_folds = 5 if "5" in self.config.cv_method else 10
        
        for clf_type in classifiers:
            logger.info(f"\n訓練 {clf_type.value}...")
            
            try:
                # 分類器配置
                clf_config = ClassifierConfig.get_default(clf_type)
                
                # 特徵選擇配置
                method = FeatureSelector.get_method_for_classifier(clf_type.value)
                feat_config = SelectionConfig(method=method) if method != SelectionMethod.NONE else None
                
                # CV配置
                cv_config = CVConfig(
                    method=cv_method,
                    n_folds=n_folds,
                    feature_selection=feat_config
                )
                
                # 執行CV
                validator = CrossValidator(cv_config)
                cv_results = validator.validate(X, y, subject_ids, clf_config)
                
                # 儲存結果
                results[clf_type.value] = {
                    'accuracy': cv_results.accuracy,
                    'mcc': cv_results.mcc,
                    'sensitivity': cv_results.sensitivity,
                    'specificity': cv_results.specificity,
                    'confusion_matrix': cv_results.confusion_matrix.tolist()
                }
                
                logger.info(f"  準確率: {cv_results.accuracy:.3f}")
                logger.info(f"  MCC: {cv_results.mcc:.3f}")
                
            except Exception as e:
                logger.error(f"  {clf_type.value} 訓練失敗: {e}")
                results[clf_type.value] = {'error': str(e)}
        
        return results
    
    def run(
        self,
        data_path: str,
        demo_processor: Optional[DemographicsProcessor] = None
    ) -> Dict[str, Any]:
        """執行完整流程"""
        
        logger.info("="*60)
        logger.info("開始訓練流程")
        logger.info("="*60)
        
        all_results = {}
        
        for cdr_threshold in self.config.cdr_thresholds:
            
            # 處理CDR篩選
            if self.config.enable_cdr_filter and cdr_threshold is not None:
                logger.info(f"\n使用 CDR > {cdr_threshold} 篩選")
                # 這裡應該要篩選，但簡化版先跳過
            
            # 準備資料
            X, y, subject_ids, lookup = self.prepare_data(data_path, demo_processor)
            
            # 訓練所有分類器
            results = self.train_all_classifiers(X, y, subject_ids)
            
            # 儲存結果
            key = f"CDR_{cdr_threshold}" if cdr_threshold else "All_data"
            all_results[key] = results
        
        # 儲存到檔案
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict):
        """儲存結果"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n結果已儲存至: {filename}")
        
        # 印出摘要
        self.print_summary(results)
    
    def print_summary(self, results: Dict):
        """印出結果摘要"""
        print("\n" + "="*60)
        print("訓練結果摘要")
        print("="*60)
        
        for condition, classifiers in results.items():
            print(f"\n{condition}:")
            for clf_name, metrics in classifiers.items():
                if 'error' in metrics:
                    print(f"  {clf_name}: 錯誤 - {metrics['error'][:50]}")
                else:
                    print(f"  {clf_name}: Acc={metrics['accuracy']:.3f}, MCC={metrics['mcc']:.3f}")