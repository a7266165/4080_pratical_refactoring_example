# scripts/test_models.py
"""測試模型訓練相關功能"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.loader import FeatureDataLoader
from src.models.classifiers import ClassifierFactory, ClassifierConfig, ClassifierType
from src.train.cross_validation import CrossValidator, CVConfig, CVMethod
from src.features.selection import SelectionConfig, SelectionMethod

def test_classifier_factory():
    """測試分類器工廠"""
    print("\n" + "="*60)
    print("測試分類器工廠")
    print("="*60)
    
    factory = ClassifierFactory()
    
    # 測試所有分類器類型
    for classifier_type in ClassifierType:
        print(f"\n創建 {classifier_type.value}:")
        
        # 使用預設配置
        config = ClassifierConfig.get_default(classifier_type)
        classifier = factory.create_classifier(config)
        
        print(f"  類型: {type(classifier).__name__}")
        print(f"  需要標準化: {config.need_scaling}")
        print(f"  參數: {config.params}")
        
        # 簡單的訓練測試
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 20)
        
        # 準備資料
        X_train_p, X_test_p, _ = factory.prepare_data(X_train, X_test, config)
        
        # 訓練
        classifier.fit(X_train_p, y_train)
        y_pred = classifier.predict(X_test_p)
        
        print(f"  預測形狀: {y_pred.shape}")
        print(f"  ✓ {classifier_type.value} 測試通過")
    
    print("\n✓ 分類器工廠測試完成")

def test_cross_validation():
    """測試交叉驗證"""
    print("\n" + "="*60)
    print("測試交叉驗證")
    print("="*60)
    
    # 建立測試資料
    np.random.seed(42)
    n_samples = 150
    n_features = 20
    n_subjects = 30
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # 建立受試者ID（每個受試者有多個樣本）
    subject_ids = []
    samples_per_subject = n_samples // n_subjects
    for i in range(n_subjects):
        subject_ids.extend([f"S{i:03d}"] * samples_per_subject)
    # 處理剩餘樣本
    for i in range(n_samples - len(subject_ids)):
        subject_ids.append(f"S{i:03d}")
    
    # 測試 LOSO
    print("\n測試 Leave-One-Subject-Out:")
    cv_config = CVConfig(method=CVMethod.LOSO)
    validator = CrossValidator(cv_config)
    
    classifier_config = ClassifierConfig.get_default(ClassifierType.RANDOM_FOREST)
    results = validator.validate(X, y, subject_ids, classifier_config)
    
    print(f"  受試者數: {n_subjects}")
    results.print_summary()
    
    # 測試 K-Fold
    print("\n測試 5-Fold 交叉驗證:")
    cv_config = CVConfig(method=CVMethod.KFOLD, n_folds=5)
    validator = CrossValidator(cv_config)
    
    results = validator.validate(X, y, subject_ids, classifier_config)
    results.print_summary()
    
    print("\n✓ 交叉驗證測試完成")

def test_integrated_pipeline():
    """測試整合的訓練流程"""
    print("\n" + "="*60)
    print("測試整合訓練流程")
    print("="*60)
    
    # 建立測試資料
    np.random.seed(42)
    X = np.random.randn(200, 50)
    # 建立有意義的標籤（基於前幾個特徵）
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    subject_ids = [f"S{i//5:03d}" for i in range(200)]  # 每個受試者5個樣本
    
    # 測試配置組合
    test_cases = [
        ("RF + LOSO", 
         ClassifierType.RANDOM_FOREST, 
         CVMethod.LOSO,
         None),
        
        ("XGB + 5-Fold + 特徵選擇",
         ClassifierType.XGB,
         CVMethod.KFOLD,
         SelectionConfig(method=SelectionMethod.XGB_IMPORTANCE, importance_ratio=0.5)),
        
        ("SVM + LOSO + 相關性過濾",
         ClassifierType.SVM,
         CVMethod.LOSO,
         SelectionConfig(method=SelectionMethod.CORRELATION, correlation_threshold=0.8)),
    ]
    
    print("\n測試不同配置組合:")
    print("-" * 50)
    
    for name, clf_type, cv_method, feat_config in test_cases:
        print(f"\n{name}:")
        
        # 配置
        classifier_config = ClassifierConfig.get_default(clf_type)
        cv_config = CVConfig(
            method=cv_method,
            n_folds=5 if cv_method == CVMethod.KFOLD else None,
            feature_selection=feat_config
        )
        
        # 執行
        validator = CrossValidator(cv_config)
        results = validator.validate(X, y, subject_ids, classifier_config)
        
        # 顯示結果
        print(f"  準確率: {results.accuracy:.3f}")
        print(f"  MCC: {results.mcc:.3f}")
    
    print("\n✓ 整合流程測試完成")

def main():
    """執行所有測試"""
    print("\n" + "#"*60)
    print("# 模型訓練模組測試")
    print("#"*60)
    
    # 測試各個組件
    test_classifier_factory()
    test_cross_validation()
    test_integrated_pipeline()
    
    print("\n" + "#"*60)
    print("# 所有測試完成！")
    print("#"*60)

if __name__ == '__main__':
    main()