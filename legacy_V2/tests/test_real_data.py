# tests/test_real_data.py
"""測試實際資料載入與處理"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.data.loader import FeatureDataLoader
from src.data.demographics import DemographicsProcessor, DemographicsConfig
from src.data.balancing import DataBalancer, BalancingConfig
from src.features.selection import SelectionConfig, SelectionMethod
from src.models.classifiers import ClassifierConfig, ClassifierType
from src.train.cross_validation import CrossValidator, CVConfig, CVMethod

# 實際資料路徑
DATA_PATHS = {
    'features': r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature_V2\datung",
    'p_csv': r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\p_merged.csv",
    'acs_csv': r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\ACS_merged_results.csv", 
    'nad_csv': r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\NAD_merged_results.csv"
}

def test_data_loading():
    """測試特徵資料載入"""
    print("\n" + "="*60)
    print("1. 測試特徵資料載入")
    print("="*60)
    
    try:
        loader = FeatureDataLoader(DATA_PATHS['features'])
        
        # 測試掃描個案
        print("\n掃描所有個案（最新訪視）...")
        subjects_latest = loader.scan_subjects(use_all_visits=False)
        print(f"  找到 {len(subjects_latest)} 個個案（最新訪視）")
        
        # 測試掃描所有訪視
        print("\n掃描所有個案（所有訪視）...")
        subjects_all = loader.scan_subjects(use_all_visits=True)
        print(f"  找到 {len(subjects_all)} 個樣本（所有訪視）")
        
        # 統計各組人數
        group_counts = {}
        for subject in subjects_latest:
            group_counts[subject.group] = group_counts.get(subject.group, 0) + 1
        
        print("\n各組人數（最新訪視）:")
        for group, count in sorted(group_counts.items()):
            print(f"  {group}: {count} 人")
        
        # 測試載入特徵
        print("\n測試載入VGGFace差異特徵...")
        feature_data = loader.load_features(
            subjects_latest[:5],  # 只測試前5個
            embedding_model='vggface',
            feature_type='difference'
        )
        
        print(f"  成功載入 {len(feature_data)} 個個案的特徵")
        if feature_data:
            first_feature = list(feature_data[0].features.values())[0]
            print(f"  特徵維度: {first_feature.shape}")
        
        # 取得資料集資訊
        info = loader.get_dataset_info(feature_data)
        print(f"\n資料集資訊:")
        print(f"  樣本數: {info.n_samples}")
        print(f"  健康組: {info.n_health}")
        print(f"  病患組: {info.n_patient}")
        print(f"  特徵維度: {info.feature_dim}")
        
        print("\n✓ 特徵資料載入測試通過")
        return loader, subjects_latest
        
    except Exception as e:
        print(f"\n✗ 特徵資料載入失敗: {e}")
        return None, None

def test_demographics_loading():
    """測試人口學資料載入"""
    print("\n" + "="*60)
    print("2. 測試人口學資料載入")
    print("="*60)
    
    try:
        # 初始化處理器
        config = DemographicsConfig(use_latest_visit=True)
        processor = DemographicsProcessor(config)
        
        # 載入年齡表
        print("\n載入年齡表...")
        tables = processor.load_tables(
            p_source=DATA_PATHS['p_csv'],
            acs_source=DATA_PATHS['acs_csv'],
            nad_source=DATA_PATHS['nad_csv']
        )
        
        print("\n各組資料筆數:")
        for group, df in tables.items():
            print(f"  {group}: {len(df)} 筆")
            
            # 顯示年齡統計
            age_stats = df['Age'].describe()
            print(f"    年齡: {age_stats['mean']:.1f}±{age_stats['std']:.1f} "
                  f"({age_stats['min']:.0f}-{age_stats['max']:.0f})")
            
            # 檢查CDR欄位
            if 'Global_CDR' in df.columns:
                cdr_stats = df['Global_CDR'].describe()
                print(f"    CDR: {cdr_stats['mean']:.2f}±{cdr_stats['std']:.2f} "
                      f"({cdr_stats['min']:.1f}-{cdr_stats['max']:.1f})")
            
            # 檢查性別欄位
            if 'Sex' in df.columns:
                sex_counts = df['Sex'].value_counts()
                print(f"    性別分布: {dict(sex_counts)}")
        
        # 建立查詢表
        print("\n建立人口學查詢表...")
        lookup = processor.build_lookup_table()
        print(f"  查詢表大小: {len(lookup)} 筆")
        
        # 測試查詢
        test_ids = ["P1", "ACS1", "NAD1"]
        print("\n測試查詢:")
        for test_id in test_ids:
            if test_id in lookup:
                data = lookup[test_id]
                print(f"  {test_id}: Age={data['Age']:.0f}, Sex={data['Sex']}")
        
        print("\n✓ 人口學資料載入測試通過")
        return processor
        
    except FileNotFoundError as e:
        print(f"\n✗ 找不到檔案: {e}")
        print("  請確認年齡表路徑是否正確")
        return None
    except Exception as e:
        print(f"\n✗ 人口學資料載入失敗: {e}")
        return None

def test_data_balancing(processor):
    """測試資料平衡"""
    print("\n" + "="*60)
    print("3. 測試資料平衡策略")
    print("="*60)
    
    if processor is None:
        print("  跳過（需要先載入人口學資料）")
        return None
    
    try:
        # 測試1: 無平衡
        print("\n測試無平衡策略...")
        config = BalancingConfig(
            enable_age_matching=False,
            enable_cdr_filter=False
        )
        balancer = DataBalancer(processor, config)
        allowed_ids, summary = balancer.balance_groups()
        
        print("  結果:")
        for _, row in summary.iterrows():
            print(f"    {row['group']}: n={row['n']:.0f}, "
                  f"age={row['age_mean']:.1f}±{row['age_std']:.1f}")
        
        # 測試2: 年齡配對
        print("\n測試年齡配對策略...")
        config = BalancingConfig(
            enable_age_matching=True,
            enable_cdr_filter=False,
            n_bins=5
        )
        balancer = DataBalancer(processor, config)
        allowed_ids_age, summary_age = balancer.balance_groups()
        
        print("  配對結果:")
        for _, row in summary_age.iterrows():
            print(f"    {row['group']}: n={row['n']:.0f}, "
                  f"age={row['age_mean']:.1f}±{row['age_std']:.1f}")
        
        # 測試3: CDR篩選
        print("\n測試CDR篩選策略...")
        config = BalancingConfig(
            enable_age_matching=False,
            enable_cdr_filter=True,
            cdr_threshold=0.5
        )
        balancer = DataBalancer(processor, config)
        allowed_ids_cdr, summary_cdr = balancer.balance_groups()
        
        print("  CDR>0.5篩選結果:")
        for _, row in summary_cdr.iterrows():
            print(f"    {row['group']}: n={row['n']:.0f}")
        
        # 測試4: CDR篩選+年齡配對
        print("\n測試CDR篩選+年齡配對...")
        config = BalancingConfig(
            enable_age_matching=True,
            enable_cdr_filter=True,
            cdr_threshold=1.0,
            n_bins=5
        )
        balancer = DataBalancer(processor, config)
        allowed_ids_both, summary_both = balancer.balance_groups()
        
        print("  CDR>1.0+年齡配對結果:")
        for _, row in summary_both.iterrows():
            print(f"    {row['group']}: n={row['n']:.0f}, "
                  f"age={row['age_mean']:.1f}±{row['age_std']:.1f}")
        
        print("\n✓ 資料平衡測試通過")
        return allowed_ids_age  # 返回年齡配對的結果供後續使用
        
    except Exception as e:
        print(f"\n✗ 資料平衡測試失敗: {e}")
        return None

def test_integrated_pipeline(loader, subjects, processor, allowed_ids):
    """測試整合流程"""
    print("\n" + "="*60)
    print("4. 測試整合訓練流程")
    print("="*60)
    
    if not all([loader, subjects, processor]):
        print("  跳過（需要先完成前面的測試）")
        return
    
    try:
        # 載入特徵資料
        print("\n載入特徵資料...")
        feature_data = loader.load_features(
            subjects,
            embedding_model='vggface',
            feature_type='difference'
        )
        
        # 準備訓練資料
        X_list = []
        y_list = []
        subject_ids = []
        
        for fd in feature_data:
            # 檢查是否在允許的ID清單中
            if allowed_ids:
                base_id = fd.subject_info.subject_id
                group = fd.subject_info.group
                if group in allowed_ids and base_id not in allowed_ids[group]:
                    continue
            
            features = list(fd.features.values())[0]
            X_list.append(features)
            y_list.append(fd.subject_info.label)
            subject_ids.append(fd.subject_info.subject_id)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"  資料形狀: {X.shape}")
        print(f"  標籤分布: 健康={np.sum(y==0)}, 病患={np.sum(y==1)}")
        print(f"  獨立受試者: {len(set(subject_ids))}")
        
        # 測試不同的訓練配置
        test_configs = [
            ("RF + LOSO", ClassifierType.RANDOM_FOREST, CVMethod.LOSO, None),
            ("XGB + 5-Fold + 特徵選擇", ClassifierType.XGB, CVMethod.KFOLD,
             SelectionConfig(method=SelectionMethod.XGB_IMPORTANCE, importance_ratio=0.8)),
        ]
        
        print("\n測試不同訓練配置:")
        print("-" * 50)
        
        for name, clf_type, cv_method, feat_config in test_configs:
            print(f"\n{name}:")
            
            # 配置
            classifier_config = ClassifierConfig.get_default(clf_type)
            cv_config = CVConfig(
                method=cv_method,
                n_folds=5 if cv_method == CVMethod.KFOLD else None,
                feature_selection=feat_config
            )
            
            # 執行交叉驗證（只用部分資料以加快速度）
            validator = CrossValidator(cv_config)
            
            # 限制資料量以加快測試
            n_samples = min(100, len(X))
            X_subset = X[:n_samples]
            y_subset = y[:n_samples]
            subject_ids_subset = subject_ids[:n_samples]
            
            results = validator.validate(
                X_subset, y_subset, subject_ids_subset,
                classifier_config
            )
            
            # 顯示結果
            results.print_summary()
        
        print("\n✓ 整合流程測試通過")
        
    except Exception as e:
        print(f"\n✗ 整合流程測試失敗: {e}")
        import traceback
        traceback.print_exc()

def main():
    """執行所有測試"""
    print("\n" + "#"*60)
    print("# 實際資料載入與處理測試")
    print("#"*60)
    
    # 1. 測試特徵資料載入
    loader, subjects = test_data_loading()
    
    # 2. 測試人口學資料載入
    processor = test_demographics_loading()
    
    # 3. 測試資料平衡
    allowed_ids = None
    if processor:
        allowed_ids = test_data_balancing(processor)
    
    # 4. 測試整合流程
    test_integrated_pipeline(loader, subjects, processor, allowed_ids)
    
    print("\n" + "#"*60)
    print("# 測試完成！")
    print("#"*60)
    
    # 總結
    print("\n測試總結:")
    print("-" * 40)
    print(f"✓ 特徵資料載入: {'成功' if loader else '失敗'}")
    print(f"✓ 人口學資料載入: {'成功' if processor else '失敗'}")
    print(f"✓ 資料平衡: {'成功' if allowed_ids else '失敗'}")

if __name__ == '__main__':
    main()