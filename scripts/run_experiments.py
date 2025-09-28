# scripts/run_experiments.py
"""系統化實驗：測試不同資料平衡策略的效果"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from datetime import datetime
from itertools import product
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import FeatureDataLoader
from src.data.demographics import DemographicsProcessor
from src.data.balancing import DataBalancer, BalancingConfig
from src.models.classifiers import ClassifierFactory, ClassifierConfig, ClassifierType
from src.train.cross_validation import CrossValidator, CVConfig, CVMethod
from src.features.selection import FeatureSelector, SelectionConfig, SelectionMethod
from src.utils.id_parser import parse_subject_id

# ============= 實驗配置 =============
DATA_PATH = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature_V2\datung"
P_CSV = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\p_merged.csv"
ACS_CSV = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\ACS_merged_results.csv"
NAD_CSV = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\NAD_merged_results.csv"

# 實驗矩陣
EXPERIMENTS = [
    {
        'name': '基準線（無平衡）',
        'age_matching': False,
        'cdr_filter': False,
        'cdr_thresholds': [None]
    },
    {
        'name': '年齡配對',
        'age_matching': True,
        'cdr_filter': False,
        'cdr_thresholds': [None]
    },
    {
        'name': '年齡配對 + CDR篩選',
        'age_matching': True,
        'cdr_filter': True,
        'cdr_thresholds': [0.5, 1.0, 2.0]  # 可以測試多個閾值
    }
]

# 要測試的嵌入模型
EMBEDDING_MODELS = ['vggface', 'arcface', 'dlib', 'deepid', 'topofr']

# 要測試的特徵類型
FEATURE_TYPES = ['difference', 'average', 'relative']

# 要測試的分類器
CLASSIFIERS = [
    ClassifierType.RANDOM_FOREST,
    ClassifierType.XGB,
    # ClassifierType.SVM,  # SVM可能較慢，可選
    ClassifierType.LOGISTIC
]

# =====================================


class ExperimentRunner:
    """實驗執行器"""
    
    def __init__(self):
        self.results = {}
        self.loader = None
        self.demo_processor = None
        
    def setup(self):
        """初始化資料載入器"""
        print("="*60)
        print("初始化實驗環境")
        print("="*60)
        
        # 載入特徵資料載入器
        self.loader = FeatureDataLoader(DATA_PATH)
        subjects = self.loader.scan_subjects(use_all_visits=False)
        print(f"✓ 載入特徵資料：{len(subjects)} 個個案")
        
        # 載入人口學資料
        try:
            self.demo_processor = DemographicsProcessor()
            self.demo_processor.load_tables(
                p_source=P_CSV,
                acs_source=ACS_CSV,
                nad_source=NAD_CSV
            )
            print(f"✓ 載入人口學資料")
        except Exception as e:
            print(f"⚠ 人口學資料載入失敗: {e}")
            self.demo_processor = None
    
    def prepare_data(
        self,
        embedding_model: str,
        feature_type: str,
        balance_config: BalancingConfig
    ) -> tuple:
        """準備訓練資料"""
        
        # 掃描個案
        subjects = self.loader.scan_subjects(use_all_visits=False)

        # # 資料平衡
        # allowed_ids = None
        # if balance_config and self.demo_processor:
        #     balancer = DataBalancer(self.demo_processor, balance_config)
        #     allowed_ids, summary = balancer.balance_groups()
        #     print(f" 平衡後：", end="")
        #     for _, row in summary.iterrows():
        #         print(f"{row['group']}={int(row['n'])} ", end="")
        #     print()

        # 資料平衡
        import traceback

        DEBUG_BALANCE = True  # 想關閉除錯輸出時改成 False

        allowed_ids = None
        if balance_config and self.demo_processor:
            if DEBUG_BALANCE:
                print("\n[DBG] === 開始資料平衡 ===")
                print(f"[DBG] balance_config: {balance_config!r}")
                print(f"[DBG] demo_processor: {type(self.demo_processor).__name__}")

                # 嘗試檢視 tables 狀態與基本統計
                try:
                    tables = getattr(self.demo_processor, "tables", {})
                    print(f"[DBG] tables.keys(): {list(tables.keys())}")

                    for k in ("ACS", "NAD", "P"):
                        df = tables.get(k, None)
                        print(
                            f"[DBG] {k}: is None? {df is None}, "
                            f"shape={getattr(df, 'shape', None)}, "
                            f"cols={list(df.columns) if hasattr(df, 'columns') else None}"
                        )
                        if df is not None and hasattr(df, "columns"):
                            if "ID" in df.columns:
                                try:
                                    print(f"[DBG] {k} sample IDs: {df['ID'].head(5).tolist()}")
                                except Exception as _:
                                    pass
                            if "Age" in df.columns:
                                try:
                                    print(
                                        f"[DBG] {k} Age stats: "
                                        f"n={len(df)}, "
                                        f"min={df['Age'].min()}, "
                                        f"max={df['Age'].max()}, "
                                        f"mean={df['Age'].mean():.2f}"
                                    )
                                except Exception as _:
                                    pass
                except Exception as e:
                    print("[DBG][ERROR] 讀取 demo_processor.tables 時發生例外：", e)
                    traceback.print_exc()

            try:
                balancer = DataBalancer(self.demo_processor, balance_config)
                if DEBUG_BALANCE:
                    print(f"[DBG] 建立 DataBalancer 成功：{balancer!r}")

                allowed_ids, summary = balancer.balance_groups()

            except Exception as e:
                print("[ERR] balancer.balance_groups() 拋出例外：", e)
                traceback.print_exc()
                # 視需求可選擇 raise 或回傳預設
                raise

            # 呼叫後檢查輸出
            if DEBUG_BALANCE:
                print("[DBG] allowed_ids 類型：", type(allowed_ids))
                if isinstance(allowed_ids, dict):
                    for g in ("ACS", "NAD", "P"):
                        s = allowed_ids.get(g, None)
                        size = (len(s) if isinstance(s, (set, list)) else "N/A")
                        sample = (list(s)[:5] if isinstance(s, set) else "N/A")
                        print(f"[DBG] allowed_ids[{g}]: size={size}, sample={sample}")

                print("[DBG] summary 型態/屬性：",
                    f"has iterrows? {hasattr(summary, 'iterrows')}, "
                    f"shape={getattr(summary, 'shape', None)}, "
                    f"columns={getattr(summary, 'columns', None)}")

                # 若是 DataFrame 顯示前幾列
                try:
                    if hasattr(summary, "head"):
                        print("[DBG] summary.head():")
                        print(summary.head())
                except Exception as _:
                    pass

            # 使用更穩健的輸出，避免 summary 為 None 或缺欄位時崩潰
            print(f"    平衡後：", end="")
            try:
                is_df_like = hasattr(summary, "iterrows") and hasattr(summary, "columns")
                if is_df_like and not getattr(summary, "empty", False) \
                and "group" in summary.columns and "n" in summary.columns:
                    for _, row in summary.iterrows():
                        try:
                            print(f"{row['group']}={int(row['n'])} ", end="")
                        except Exception:
                            print(f"{row.get('group','?')}={row.get('n','?')} ", end="")
                else:
                    print(f"[DBG] summary 非預期（空或缺欄位）：{type(summary)}", end="")
            finally:
                print()
        


        # 載入特徵
        feature_data = self.loader.load_features(
            subjects,
            embedding_model=embedding_model,
            feature_type=feature_type
        )

        # 篩選允許的ID（如果有平衡）
        # if allowed_ids:
        #     filtered_data = []
        #     for fd in feature_data:
        #         subject_id = fd.subject_info.subject_id
        #         base_id, _ = parse_subject_id(subject_id)
        #         group = fd.subject_info.group
                
        #         # 檢查是否在允許清單中
        #         if group in allowed_ids:
        #             if subject_id in allowed_ids[group] or base_id in allowed_ids[group]:
        #                 filtered_data.append(fd)
            
        #     feature_data = filtered_data
        #     print(f"    篩選後: {len(feature_data)} 筆")
        # 篩選允許的ID（如果有平衡）
        if allowed_ids is not None:
            print("[DBG] === 開始篩選允許的 ID ===")

            from collections import Counter
            # 1) 觀察 feature_data 的群組分佈
            grp_counts = Counter(getattr(fd.subject_info, "group", None) for fd in feature_data)
            print(f"[DBG] feature_data groups: {dict(grp_counts)} (total={len(feature_data)})")

            # 2) 建立 allowed_ids 的「base_id」快取（把 allowed 的 ID 也做同樣的 parse）
            def to_base_set(id_set):
                bs = set()
                for x in id_set:
                    try:
                        b, _ = parse_subject_id(x)
                    except Exception:
                        b = str(x)
                    bs.add(b)
                return bs

            allowed_base_cache = {k: to_base_set(v) for k, v in allowed_ids.items()}

            print("[DBG] allowed sizes:", {k: len(v) for k, v in allowed_ids.items()})
            print("[DBG] allowed(base) sizes:", {k: len(v) for k, v in allowed_base_cache.items()})

            # 3) 把 Health/Healthy 對應到 ACS+NAD；Patient 對應到 P
            def expand_keys_by_group(grp: str):
                if grp in ("Health", "Healthy", "Control"):
                    return ["ACS", "NAD"]
                if grp in ("Patient",):
                    return ["P"]
                # 已是 ACS/NAD/P 或其他自定義
                return [grp]

            # 4) 實際比對 + 列印前幾個 miss 範例
            miss_samples = 0
            filtered_data = []

            for i, fd in enumerate(feature_data):
                sid = getattr(fd.subject_info, "subject_id", None)
                grp = getattr(fd.subject_info, "group", None)
                if sid is None or grp is None:
                    if miss_samples < 5:
                        print(f"[DBG][miss] 欄位缺失 grp={grp} sid={sid}")
                        miss_samples += 1
                    continue

                try:
                    base, _ = parse_subject_id(sid)
                except Exception:
                    base = sid  # 解析失敗就當作原字串

                keys = expand_keys_by_group(grp)

                matched = False
                for key in keys:
                    if key not in allowed_ids:
                        continue
                    # 先比完整 ID，再比 base ID
                    if sid in allowed_ids[key] or base in allowed_base_cache[key]:
                        matched = True
                        break

                if matched:
                    filtered_data.append(fd)
                elif miss_samples < 5:
                    print(f"[DBG][miss] grp={grp} sid={sid} base={base} keys={keys}")
                    for key in keys:
                        if key in allowed_ids:
                            print(f"      allowed[{key}]: sid_hit={sid in allowed_ids[key]}, "
                                f"base_hit={base in allowed_base_cache[key]}")
                    miss_samples += 1

            print(f"    篩選後: {len(filtered_data)} 筆 (原始 {len(feature_data)})")
            feature_data = filtered_data

            # 5) 額外的交集統計，快速判斷問題在「群組」還是「ID 正規化」
            try:
                # 取出 feature_data 中，各群組的 sid/base 集合
                feat_sid = {}
                feat_base = {}
                for fd in feature_data:
                    g = getattr(fd.subject_info, "group", None)
                    s = getattr(fd.subject_info, "subject_id", None)
                    if g is None or s is None: 
                        continue
                    b, _ = parse_subject_id(s)
                    feat_sid.setdefault(g, set()).add(s)
                    feat_base.setdefault(g, set()).add(b)

                # 用 Health→ACS+NAD 的對映做交集統計
                for g in grp_counts.keys():
                    keys = expand_keys_by_group(g)
                    union_allowed_sid = set().union(*(allowed_ids.get(k, set()) for k in keys))
                    union_allowed_base = set().union(*(allowed_base_cache.get(k, set()) for k in keys))
                    print(f"[DBG] intersect {g}: "
                        f"sid={len(feat_sid.get(g, set()) & union_allowed_sid)}, "
                        f"base={len(feat_base.get(g, set()) & union_allowed_base)}")
            except Exception as _:
                pass
        
        
        # 準備特徵矩陣
        X_list = []
        y_list = []
        subject_ids = []
        
        for fd in feature_data:
            features = list(fd.features.values())[0]
            if features is not None:
                X_list.append(features)
                y_list.append(fd.subject_info.label)
                subject_ids.append(fd.subject_info.subject_id)
        
        if not X_list:
            return None, None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 添加人口學特徵
        if self.demo_processor:
            X = self._add_demographics(X, subject_ids)
        
        return X, y, subject_ids
    
    def _add_demographics(self, X: np.ndarray, subject_ids: List[str]) -> np.ndarray:
        """添加人口學特徵"""
        lookup = self.demo_processor.lookup_table
        if not lookup:
            lookup = self.demo_processor.build_lookup_table()
        
        demo_features = []
        for sid in subject_ids:
            meta = lookup.get(sid)
            if meta is None:
                base_id, _ = parse_subject_id(sid)
                meta = lookup.get(base_id)
            
            if meta:
                age = meta.get('Age', 70)
                sex = meta.get('Sex', 0.5)
            else:
                age = 70
                sex = 0.5
            
            demo_features.append([age, sex])
        
        demo_array = np.array(demo_features)
        
        # 標準化並結合
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_demo = StandardScaler()
        
        # 處理常數特徵問題
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = scaler_X.fit_transform(X)
            demo_scaled = scaler_demo.fit_transform(demo_array)
        
        X_combined = np.hstack([X_scaled, demo_scaled])
        
        return X_combined
    
    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        classifier_type: ClassifierType
    ) -> Dict[str, float]:
        """訓練單個分類器"""
        
        # 配置
        clf_config = ClassifierConfig.get_default(classifier_type)
        
        # 特徵選擇（根據分類器類型）
        if classifier_type in [ClassifierType.SVM, ClassifierType.LOGISTIC]:
            feat_method = SelectionMethod.CORRELATION
        else:
            feat_method = SelectionMethod.NONE  # 樹模型不需要特徵選擇
        
        feat_config = SelectionConfig(
            method=feat_method,
            correlation_threshold=0.95
        ) if feat_method != SelectionMethod.NONE else None
        
        # CV配置
        cv_config = CVConfig(
            method=CVMethod.KFOLD,
            n_folds=5,
            feature_selection=feat_config
        )
        
        # 執行CV
        try:
            validator = CrossValidator(cv_config)
            results = validator.validate(X, y, subject_ids, clf_config)
            
            return {
                'accuracy': results.accuracy,
                'mcc': results.mcc,
                'sensitivity': results.sensitivity,
                'specificity': results.specificity
            }
        except Exception as e:
            print(f"      錯誤: {e}")
            return {
                'accuracy': 0,
                'mcc': 0,
                'sensitivity': 0,
                'specificity': 0,
                'error': str(e)
            }
    
    def run_experiment(self, exp_config: Dict) -> Dict:
        """執行單個實驗配置"""
        print(f"\n{'='*60}")
        print(f"實驗：{exp_config['name']}")
        print(f"{'='*60}")
        
        exp_results = {}
        
        # 對每個CDR閾值
        for cdr_threshold in exp_config['cdr_thresholds']:
            
            if exp_config['cdr_filter'] and cdr_threshold is not None:
                print(f"\nCDR > {cdr_threshold}:")
            
            # 建立平衡配置
            balance_config = BalancingConfig(
                enable_age_matching=exp_config['age_matching'],
                enable_cdr_filter=exp_config['cdr_filter'],
                cdr_threshold=cdr_threshold
            )
            
            threshold_results = {}
            
            # 對每個嵌入模型
            for embedding_model in EMBEDDING_MODELS:
                print(f"\n  {embedding_model}:")
                
                model_results = {}
                
                # 對每個特徵類型
                for feature_type in FEATURE_TYPES:
                    
                    # 準備資料
                    X, y, subject_ids = self.prepare_data(
                        embedding_model,
                        feature_type,
                        balance_config
                    )
                    
                    if X is None:
                        print(f"    {feature_type}: 無資料")
                        continue
                    
                    print(f"    {feature_type}: {X.shape}, 健康={np.sum(y==0)}, 病患={np.sum(y==1)}")
                    
                    # 對每個分類器
                    classifier_results = {}
                    for classifier_type in CLASSIFIERS:
                        print(f"      {classifier_type.value}...", end=" ")
                        
                        result = self.train_classifier(
                            X, y, subject_ids,
                            classifier_type
                        )
                        
                        classifier_results[classifier_type.value] = result
                        print(f"Acc={result['accuracy']:.3f}, MCC={result['mcc']:.3f}")
                    
                    model_results[feature_type] = classifier_results
                
                threshold_results[embedding_model] = model_results
            
            # 儲存結果
            key = f"CDR_{cdr_threshold}" if cdr_threshold else "All"
            exp_results[key] = threshold_results
        
        return exp_results
    
    def run_all_experiments(self):
        """執行所有實驗"""
        self.setup()
        
        print("\n" + "="*60)
        print("開始實驗")
        print("="*60)
        
        all_results = {}
        
        for exp_config in EXPERIMENTS:
            exp_results = self.run_experiment(exp_config)
            all_results[exp_config['name']] = exp_results
        
        self.results = all_results
        return all_results
    
    def save_results(self, filename: str = None):
        """儲存結果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
        
        output_path = Path("results")
        output_path.mkdir(exist_ok=True)
        
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n結果已儲存至: {filepath}")
        
        # 生成摘要表格
        self.print_summary()
    
    def print_summary(self):
        """印出結果摘要表格"""
        print("\n" + "="*60)
        print("實驗結果摘要")
        print("="*60)
        
        # 收集所有結果到表格
        rows = []
        
        for exp_name, exp_results in self.results.items():
            for cdr_key, models in exp_results.items():
                for model_name, features in models.items():
                    for feat_type, classifiers in features.items():
                        for clf_name, metrics in classifiers.items():
                            rows.append({
                                '實驗': exp_name,
                                'CDR': cdr_key,
                                '模型': model_name,
                                '特徵': feat_type,
                                '分類器': clf_name,
                                'Acc': metrics.get('accuracy', 0),
                                'MCC': metrics.get('mcc', 0),
                                'Sens': metrics.get('sensitivity', 0),
                                'Spec': metrics.get('specificity', 0)
                            })
        
        if rows:
            df = pd.DataFrame(rows)
            
            # 找出最佳組合
            best_acc = df.loc[df['Acc'].idxmax()]
            best_mcc = df.loc[df['MCC'].idxmax()]
            
            print("\n最佳準確率:")
            print(f"  {best_acc['實驗']} / {best_acc['CDR']} / {best_acc['模型']} / {best_acc['分類器']}")
            print(f"  Acc={best_acc['Acc']:.3f}, MCC={best_acc['MCC']:.3f}")
            
            print("\n最佳MCC:")
            print(f"  {best_mcc['實驗']} / {best_mcc['CDR']} / {best_mcc['模型']} / {best_mcc['分類器']}")
            print(f"  Acc={best_mcc['Acc']:.3f}, MCC={best_mcc['MCC']:.3f}")
            
            # 各實驗平均表現
            print("\n各實驗平均表現:")
            exp_summary = df.groupby('實驗')[['Acc', 'MCC']].mean()
            print(exp_summary.round(3))
            
            # 各模型平均表現
            print("\n各嵌入模型平均表現:")
            model_summary = df.groupby('模型')[['Acc', 'MCC']].mean()
            print(model_summary.round(3))
            
            # 儲存詳細表格
            csv_path = Path("results") / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n詳細表格已儲存至: {csv_path}")


def main():
    """主程式"""
    runner = ExperimentRunner()
    
    # 執行所有實驗
    results = runner.run_all_experiments()
    
    # 儲存結果
    runner.save_results()
    
    print("\n" + "="*60)
    print("所有實驗完成！")
    print("="*60)


if __name__ == "__main__":
    main()