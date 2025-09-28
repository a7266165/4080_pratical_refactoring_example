# tests/test_full_integration.py
"""測試完整整合流程"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import traceback

def test_demographics_integration():
    """測試人口學資料整合"""
    print("\n測試人口學資料整合...")
    
    try:
        from src.data.demographics import DemographicsProcessor
        from src.data.balancing import DataBalancer, BalancingConfig
        
        # 載入人口學資料
        processor = DemographicsProcessor()
        tables = processor.load_tables(
            p_source=r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\p_merged.csv",
            acs_source=r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\ACS_merged_results.csv",
            nad_source=r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\NAD_merged_results.csv"
        )
        
        print(f"  ✓ 載入人口學資料: P={len(tables['P'])}, ACS={len(tables['ACS'])}, NAD={len(tables['NAD'])}")
        
        # 建立查詢表
        lookup = processor.build_lookup_table()
        print(f"  ✓ 建立查詢表: {len(lookup)} 筆")
        
        # 測試資料平衡
        config = BalancingConfig(enable_age_matching=False, enable_cdr_filter=False)
        balancer = DataBalancer(processor, config)
        allowed_ids, summary = balancer.balance_groups()
        
        print(f"  ✓ 資料平衡完成")
        return processor, allowed_ids, lookup
        
    except FileNotFoundError as e:
        print(f"  ⚠ 找不到人口學資料檔案: {e}")
        return None, None, None
    except Exception as e:
        print(f"  ✗ 人口學整合失敗: {e}")
        traceback.print_exc()
        return None, None, None

def test_feature_with_demographics():
    """測試特徵與人口學資料結合"""
    print("\n測試特徵與人口學資料結合...")
    
    # 先測試人口學資料
    processor, allowed_ids, lookup = test_demographics_integration()
    
    try:
        from src.data.loader import FeatureDataLoader
        
        # 載入特徵
        data_path = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature_V2\datung"
        loader = FeatureDataLoader(data_path)
        
        subjects = loader.scan_subjects(use_all_visits=False)
        print(f"  找到 {len(subjects)} 個個案")
        
        # 載入較多資料
        feature_data = loader.load_features(
            subjects,
            embedding_model='vggface',
            feature_type='difference'
        )
        print(f"  載入 {len(feature_data)} 名個案的特徵")
        
        # 準備資料（含人口學特徵）
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
        
        print(f"  原始特徵: X={X.shape}, y={y.shape}")
        print(f"  標籤分布: 健康={np.sum(y==0)}, 病患={np.sum(y==1)}")
        
        # 如果有人口學資料，嘗試整合
        if lookup:
            print("\n  嘗試整合人口學特徵...")
            demo_features = []
            
            for sid in subject_ids:
                # 嘗試查詢
                if sid in lookup:
                    meta = lookup[sid]
                else:
                    # 嘗試base_id
                    from src.utils.id_parser import parse_subject_id
                    base_id, _ = parse_subject_id(sid)
                    meta = lookup.get(base_id, None)
                
                if meta:
                    age = meta.get('Age', np.nan)
                    sex = meta.get('Sex', np.nan)
                    demo_features.append([age, sex])
                else:
                    demo_features.append([np.nan, np.nan])
            
            demo_array = np.array(demo_features)
            
            # 填補缺失值
            age_mean = np.nanmean(demo_array[:, 0])
            sex_mode = np.nanmean(demo_array[:, 1])
            
            demo_array[np.isnan(demo_array[:, 0]), 0] = age_mean if not np.isnan(age_mean) else 70
            demo_array[np.isnan(demo_array[:, 1]), 1] = sex_mode if not np.isnan(sex_mode) else 0.5
            
            print(f"  人口學特徵: {demo_array.shape}")
            
            # 標準化並結合
            from sklearn.preprocessing import StandardScaler
            
            scaler_X = StandardScaler()
            scaler_demo = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            demo_scaled = scaler_demo.fit_transform(demo_array)
            
            X_combined = np.hstack([X_scaled, demo_scaled])
            print(f"  ✓ 結合後特徵: {X_combined.shape}")
            
            return X_combined, y, subject_ids
        else:
            print("  ⚠ 無人口學資料，使用原始特徵")
            return X, y, subject_ids
            
    except Exception as e:
        print(f"  ✗ 特徵結合失敗: {e}")
        traceback.print_exc()
        return None, None, None

def test_full_cv():
    """測試完整的交叉驗證流程"""
    print("\n" + "="*60)
    print("測試完整交叉驗證流程")
    print("="*60)
    
    # 準備資料
    X, y, subject_ids = test_feature_with_demographics()
    
    if X is None or len(X) < 10:
        print("\n資料不足，無法進行交叉驗證")
        return
    
    try:
        from src.models.classifiers import ClassifierConfig, ClassifierType
        from src.train.cross_validation import CrossValidator, CVConfig, CVMethod
        from src.features.selection import SelectionConfig, SelectionMethod
        
        # 測試配置
        test_cases = [
            ("RF + 5-Fold", ClassifierType.RANDOM_FOREST, CVMethod.KFOLD, None),
            ("XGB + 5-Fold + 特徵選擇", ClassifierType.XGB, CVMethod.KFOLD,
             SelectionConfig(method=SelectionMethod.CORRELATION, correlation_threshold=0.95)),
        ]
        
        print("\n執行交叉驗證測試:")
        print("-" * 40)
        
        for name, clf_type, cv_method, feat_config in test_cases:
            print(f"\n{name}:")
            
            try:
                # 設定
                clf_config = ClassifierConfig.get_default(clf_type)
                cv_config = CVConfig(
                    method=cv_method,
                    n_folds=5,
                    feature_selection=feat_config
                )
                
                # 執行
                validator = CrossValidator(cv_config)
                results = validator.validate(X, y, subject_ids, clf_config)
                
                # 顯示結果
                print(f"  準確率: {results.accuracy:.3f}")
                print(f"  MCC: {results.mcc:.3f}")
                print(f"  靈敏度: {results.sensitivity:.3f}")
                print(f"  特異度: {results.specificity:.3f}")
                
            except Exception as e:
                print(f"  ✗ 失敗: {e}")
                traceback.print_exc()
        
    except Exception as e:
        print(f"\n✗ 交叉驗證測試失敗: {e}")
        traceback.print_exc()

def main():
    print("="*60)
    print("完整整合測試")
    print("="*60)
    
    # 執行完整測試
    test_full_cv()
    
    print("\n" + "="*60)
    print("測試結束")
    print("="*60)

if __name__ == '__main__':
    main()