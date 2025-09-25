import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import joblib
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

def parse_case_id(folder_name: str) -> Tuple[str, int]:
    """
    解析資料夾名稱，提取個案ID和收案次數
    例如: P1-2 -> (P1, 2), ACS1-1 -> (ACS1, 1)
    """
    patterns = [
        r'^([A-Za-z]+\d+)-(\d+)$',
        r'^([A-Za-z]+)-(\d+)$',
        r'^(\w+)-(\d+)$'
    ]
    for pattern in patterns:
        match = re.match(pattern, folder_name)
        if match:
            case_id = match.group(1)
            visit_number = int(match.group(2))
            return case_id, visit_number
    return folder_name, 0

def filter_latest_cases(case_folders: List[Path]) -> List[Path]:
    """
    過濾出每個個案的最後一次收案資料
    """
    case_dict = {}
    for folder in case_folders:
        folder_name = folder.name
        case_id, visit_number = parse_case_id(folder_name)
        if case_id not in case_dict:
            case_dict[case_id] = []
        case_dict[case_id].append((visit_number, folder))
    latest_folders = []
    for case_id, visits in case_dict.items():
        visits.sort(key=lambda x: x[0], reverse=True)
        latest_folders.append(visits[0][1])
    return latest_folders

def get_all_cases(case_folders: List[Path]) -> List[Path]:
    """
    獲取所有個案的所有收案資料（不過濾）
    """
    return case_folders

def read_difference_json(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_features_from_case(case_folder: Path, embedding_model: str, feature_type: str = "difference") -> Optional[np.ndarray]:
    """
    從個案資料夾提取指定類型的特徵
    feature_type: 'difference', 'average', 'relative'
    """
    json_files = sorted(case_folder.glob("*_LR_difference.json"))
    if not json_files:
        return None
    
    vectors = []
    for json_file in json_files:
        data = read_difference_json(json_file)
        
        if feature_type == "difference":
            feat = data.get("embedding_differences", {}).get(embedding_model)
        elif feature_type == "average":
            feat = data.get("embedding_averages", {}).get(embedding_model)
        elif feature_type == "relative":
            feat = data.get("relative_differences", {}).get(embedding_model)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
        if feat is not None:
            v = np.asarray(feat, dtype=float)
            vectors.append(v)
    
    if not vectors:
        return None
    
    shapes = {v.shape for v in vectors}
    if len(shapes) > 1:
        raise ValueError(f"向量長度不一致，無法取平均：{sorted(shapes)}")
    stacked = np.vstack(vectors)
    mean_vec = stacked.mean(axis=0)
    return mean_vec

def remove_highly_correlated_features(X: np.ndarray, threshold: float = 0.9) -> Tuple[np.ndarray, List[int]]:
    """
    移除高度相關的特徵（相關係數 > threshold）
    返回處理後的特徵矩陣和保留的特徵索引
    """
    if X.shape[1] <= 1:
        return X, list(range(X.shape[1]))
    
    corr_matrix = np.corrcoef(X.T)
    keep_features = []
    removed_features = set()
    
    for i in range(corr_matrix.shape[0]):
        if i in removed_features:
            continue
        keep_features.append(i)
        for j in range(i+1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > threshold:
                removed_features.add(j)
    
    print(f"    原始特徵數: {X.shape[1]}, 移除 {len(removed_features)} 個高度相關特徵, 保留 {len(keep_features)} 個")
    
    X_filtered = X[:, keep_features]
    return X_filtered, keep_features

def select_features_by_xgb_importance(X_train: np.ndarray, y_train: np.ndarray, 
                                     X_test: np.ndarray, importance_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    使用XGBoost的特徵重要性篩選特徵
    保留累積重要性達到importance_ratio的特徵
    """
    # 訓練XGBoost模型以獲取特徵重要性
    xgb_temp = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=5, n_jobs=-1)
    xgb_temp.fit(X_train, y_train)
    
    # 獲取特徵重要性（gain）
    importance = xgb_temp.feature_importances_
    
    # 排序並計算累積重要性
    indices = np.argsort(importance)[::-1]
    cumsum = np.cumsum(importance[indices])
    
    # 找出累積重要性達到閾值的特徵數量
    n_features = np.searchsorted(cumsum, importance_ratio * cumsum[-1]) + 1
    n_features = max(1, min(n_features, len(indices)))  # 至少保留1個特徵
    
    # 選擇最重要的特徵
    selected_features = indices[:n_features]
    selected_features = sorted(selected_features)  # 保持原始順序
    
    print(f"      XGBoost特徵篩選: 原始{X_train.shape[1]}個特徵 -> 保留{len(selected_features)}個特徵 (累積重要性{importance_ratio*100:.0f}%)")
    
    return X_train[:, selected_features], X_test[:, selected_features], selected_features

# ==============================
# === 年齡表與配對功能（保持不變） ===
# ==============================

def _select_latest_rows_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """將同一 base_id 只保留最後一次收案"""
    df = df.copy()
    df["base_id"] = df["ID"].apply(lambda s: parse_case_id(str(s))[0])
    df["visit_number"] = df["ID"].apply(lambda s: parse_case_id(str(s))[1])

    def pick_last(group: pd.DataFrame) -> pd.Series:
        g = group.sort_values(
            by=["visit_number", "Photo_Session"] if "Photo_Session" in group.columns else ["visit_number"],
            ascending=False
        )
        return g.iloc[0]

    latest = df.groupby("base_id", as_index=False, group_keys=False).apply(pick_last)
    latest = latest.drop(columns=["base_id", "visit_number"])
    return latest

def _read_age_table_generic(path: str) -> pd.DataFrame:
    """讀取年齡表（保持原樣）"""
    p = Path(path)

    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p)
    else:
        tried_encs = []
        df = None
        for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
            try:
                df = pd.read_csv(p, sep=None, engine="python", encoding=enc)
                break
            except Exception as e:
                tried_encs.append(f"{enc}: {e}")
        if df is None:
            raise RuntimeError(f"讀取 CSV 失敗（嘗試編碼：{'; '.join(tried_encs)}）")

    def _normalize_col(s: str) -> str:
        s = str(s)
        s = s.replace("\ufeff", "").replace("\u200b", "")
        s = s.replace("（", "(").replace("）", ")")
        s = s.strip()
        s = re.sub(r"\s+", "", s)
        return s.lower()

    raw_cols = list(df.columns)
    norm_cols = {_normalize_col(c): c for c in df.columns}

    id_keys  = {"ID"}
    age_keys = {"Age"}
    sex_keys = {"Sex"}
    cdr_keys = {"Global_CDR"}

    rename_map = {}
    found_id_src = found_age_src = found_sex_src = found_cdr_src = None

    for norm, src in norm_cols.items():
        if norm in id_keys and found_id_src is None:
            rename_map[src] = "ID"; found_id_src = src
        if norm in age_keys and found_age_src is None:
            rename_map[src] = "Age"; found_age_src = src
        if norm in sex_keys and found_sex_src is None:
            rename_map[src] = "Sex"; found_sex_src = src
        if norm in cdr_keys and found_cdr_src is None:
            rename_map[src] = "Global_CDR"; found_cdr_src = src

    df = df.rename(columns=rename_map)

    if "ID" not in df.columns or "Age" not in df.columns:
        debug_msg = (
            f"{path} 缺少必要欄位：ID 或 Age\n"
            f"實際讀到欄位：{raw_cols}\n"
            f"正規化後欄位：{[_normalize_col(c) for c in raw_cols]}"
        )
        raise ValueError(debug_msg)

    df = df.dropna(subset=["ID", "Age"])
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df.dropna(subset=["Age"])
    
    if "Global_CDR" in df.columns:
        df["Global_CDR"] = pd.to_numeric(df["Global_CDR"], errors="coerce")
        print(f"  讀取到 Global_CDR 欄位，有效值範圍: {df['Global_CDR'].min():.1f} - {df['Global_CDR'].max():.1f}")
    
    if "Sex" in df.columns:
        def _parse_sex(x):
            if pd.isna(x): return np.nan
            s = str(x).strip().lower()
            if s in {"m", "male", "man", "boy", "1", "男", "男性"}: return 1
            if s in {"f", "female", "woman", "girl", "0", "女", "女性"}: return 0
            return np.nan
        df["Sex"] = df["Sex"].apply(_parse_sex)

    return df

def load_age_tables(
    p_source: Optional[str] = None,
    acs_source: Optional[str] = None,
    nad_source: Optional[str] = None,
    excel_source: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """載入年齡表"""
    if excel_source:
        xl = pd.ExcelFile(excel_source)
        sheets = {name.lower(): name for name in xl.sheet_names}
        def pick_sheet(keyword: str) -> str:
            for k, v in sheets.items():
                if keyword.lower() in k:
                    return v
            raise ValueError(f"Excel 中找不到包含 '{keyword}' 的工作表")
        p_df = pd.read_excel(excel_source, sheet_name=pick_sheet("P"))
        acs_df = pd.read_excel(excel_source, sheet_name=pick_sheet("ACS"))
        nad_df = pd.read_excel(excel_source, sheet_name=pick_sheet("NAD"))
    else:
        if not (p_source and acs_source and nad_source):
            raise ValueError("請提供 excel_source，或同時提供 p_source、acs_source、nad_source 三個路徑")
        p_df = _read_age_table_generic(p_source)
        acs_df = _read_age_table_generic(acs_source)
        nad_df = _read_age_table_generic(nad_source)

    p_df = _select_latest_rows_by_id(p_df)
    acs_df = _select_latest_rows_by_id(acs_df)
    nad_df = _select_latest_rows_by_id(nad_df)

    return {"P": p_df, "ACS": acs_df, "NAD": nad_df}

def filter_by_cdr(tables: Dict[str, pd.DataFrame], cdr_threshold: float = None) -> Dict[str, pd.DataFrame]:
    """
    根據 Global_CDR 值篩選 P 組資料
    cdr_threshold: CDR 閾值，只保留 CDR > threshold 的資料
    """
    filtered_tables = tables.copy()
    
    if cdr_threshold is not None and "Global_CDR" in tables["P"].columns:
        original_count = len(tables["P"])
        filtered_p = tables["P"][tables["P"]["Global_CDR"] > cdr_threshold].copy()
        filtered_tables["P"] = filtered_p
        filtered_count = len(filtered_p)
        
        print(f"\n[CDR 篩選結果]")
        print(f"  CDR 閾值: > {cdr_threshold}")
        print(f"  P組原始數量: {original_count}")
        print(f"  篩選後數量: {filtered_count}")
        print(f"  保留比例: {filtered_count/original_count*100:.1f}%")
    elif cdr_threshold is not None:
        print(f"\n[警告] P 表中沒有 Global_CDR 欄位，跳過 CDR 篩選")
    
    return filtered_tables

def age_balance_ids_two_groups(
    tables: Dict[str, pd.DataFrame],
    nbins: int = 5,
    seed: int = 42,
    method: str = "quantile",
    enable_age_matching: bool = True,
    cdr_threshold: float = None
) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
    """
    兩群年齡配對（Health vs P）
    """
    rng = np.random.RandomState(seed)

    if cdr_threshold is not None:
        tables = filter_by_cdr(tables, cdr_threshold)
        if len(tables["P"]) == 0:
            print(f"  警告：CDR > {cdr_threshold} 篩選後，P 組沒有剩餘資料")
            return {"ACS": set(), "NAD": set(), "P": set()}, pd.DataFrame()

    acs_df = tables["ACS"][["ID", "Age"]].copy()
    acs_df["origin"] = "ACS"
    nad_df = tables["NAD"][["ID", "Age"]].copy()
    nad_df["origin"] = "NAD"
    p_df   = tables["P"][["ID", "Age"]].copy()

    health_df = pd.concat([acs_df, nad_df], ignore_index=True)
    health_df["group2"] = "Health"
    p_df["group2"] = "P"

    all_df = pd.concat([health_df[["ID","Age","group2","origin"]],
                        p_df[["ID","Age","group2"]]], ignore_index=True)

    if not enable_age_matching:
        print("\n[跳過年齡配對]")
        
        # 注意：年齡表中的ID可能是base_id（如P1）而不是完整ID（如P1-1）
        # 所以這裡返回的是base_id
        acs_ids = set(tables["ACS"]["ID"].tolist())
        nad_ids = set(tables["NAD"]["ID"].tolist())
        p_ids = set(tables["P"]["ID"].tolist())
        
        allowed_ids = {
            "ACS": acs_ids,
            "NAD": nad_ids,
            "P": p_ids
        }
        
        stats = []
        for g_name, ids in [("Health", acs_ids.union(nad_ids)), ("P", p_ids)]:
            g_sub = all_df[all_df["ID"].isin(ids)]
            stats.append({
                "group": g_name,
                "n_selected": int(g_sub.shape[0]),
                "age_mean": float(g_sub["Age"].mean()) if not g_sub.empty else np.nan,
                "age_std": float(g_sub["Age"].std(ddof=1)) if g_sub.shape[0] > 1 else np.nan
            })
        summary = pd.DataFrame(stats)
        
        cdr_str = f" (CDR > {cdr_threshold})" if cdr_threshold is not None else ""
        print(f"\n[統計資料（無年齡配對）{cdr_str}]")
        for _, row in summary.iterrows():
            if pd.notnull(row["age_mean"]):
                print(f"  {row['group']}: n={row['n_selected']}, mean={row['age_mean']:.2f}, std={row['age_std']:.2f}")
            else:
                print(f"  {row['group']}: n={row['n_selected']}")
        print(f"  Health 組內：ACS={len(acs_ids)}, NAD={len(nad_ids)}")
        
        return allowed_ids, summary
    
    print("\n[執行年齡配對]")
    
    if method == "quantile":
        try:
            all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins, duplicates="drop")
        except ValueError:
            nbins_eff = max(2, min(nbins, all_df["Age"].nunique()))
            all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins_eff, duplicates="drop")
    else:
        raise NotImplementedError("目前僅支援 method='quantile'")

    selected_health_ids: Set[str] = set()
    selected_p_ids: Set[str] = set()

    for b in all_df["age_bin"].cat.categories:
        bin_df = all_df[all_df["age_bin"] == b]
        n_health = bin_df[bin_df["group2"] == "Health"].shape[0]
        n_p      = bin_df[bin_df["group2"] == "P"].shape[0]
        target = min(n_health, n_p)
        if target == 0:
            continue

        health_pool = bin_df[bin_df["group2"] == "Health"]
        p_pool      = bin_df[bin_df["group2"] == "P"]

        pick_h = health_pool.sample(n=target, random_state=rng) if len(health_pool) >= target else health_pool
        pick_p = p_pool.sample(n=target, random_state=rng) if len(p_pool) >= target else p_pool

        selected_health_ids.update(pick_h["ID"].tolist())
        selected_p_ids.update(pick_p["ID"].tolist())

    acs_ids_all = set(tables["ACS"]["ID"].tolist())
    nad_ids_all = set(tables["NAD"]["ID"].tolist())

    selected_acs = set([i for i in selected_health_ids if i in acs_ids_all])
    selected_nad = set([i for i in selected_health_ids if i in nad_ids_all])

    allowed_ids = {
        "ACS": selected_acs,
        "NAD": selected_nad,
        "P":   selected_p_ids
    }

    stats = []
    for g_name, ids in [("Health", selected_health_ids), ("P", selected_p_ids)]:
        g_sub = all_df[all_df["ID"].isin(ids)]
        stats.append({
            "group": g_name,
            "n_selected": int(g_sub.shape[0]),
            "age_mean": float(g_sub["Age"].mean()) if not g_sub.empty else np.nan,
            "age_std": float(g_sub["Age"].std(ddof=1)) if g_sub.shape[0] > 1 else np.nan
        })
    summary = pd.DataFrame(stats)

    cdr_str = f" (CDR > {cdr_threshold})" if cdr_threshold is not None else ""
    print(f"\n[年齡配對後（Health vs P）的統計{cdr_str}]")
    for _, row in summary.iterrows():
        if pd.notnull(row["age_mean"]):
            print(f"  {row['group']}: n={row['n_selected']}, mean={row['age_mean']:.2f}, std={row['age_std']:.2f}")
        else:
            print(f"  {row['group']}: n={row['n_selected']}")

    print(f"  Health 組內：ACS={len(selected_acs)}, NAD={len(selected_nad)}")

    return allowed_ids, summary

def build_demo_lookup(tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """建立人口學資料查詢表"""
    frames = []
    for g in ["P", "ACS", "NAD"]:
        df = tables[g][["ID", "Age"]].copy()
        if "Sex" in tables[g].columns:
            df["Sex"] = tables[g]["Sex"]
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ID"], keep="first")

    lookup = {}
    for _, row in merged.iterrows():
        # 儲存完整ID的查詢
        lookup[str(row["ID"])] = {
            "Age": float(row["Age"]),
            "Sex": (float(row["Sex"]) if "Sex" in merged.columns and pd.notna(row["Sex"]) else np.nan)
        }
        # 同時儲存base_id的查詢（用於多次收案時）
        base_id, _ = parse_case_id(str(row["ID"]))
        if base_id not in lookup:
            lookup[base_id] = {
                "Age": float(row["Age"]),
                "Sex": (float(row["Sex"]) if "Sex" in merged.columns and pd.notna(row["Sex"]) else np.nan)
            }
    
    if "Sex" in merged.columns:
        sex_mode = merged["Sex"].dropna().mode()
        lookup["_SEX_MODE_"] = float(sex_mode.iloc[0]) if len(sex_mode)>0 else np.nan
    else:
        lookup["_SEX_MODE_"] = np.nan
    return lookup

def _norm_id(s: str) -> str:
    """統一大小寫、dash 與空白"""
    return str(s).strip().replace('－', '-').replace('—', '-').upper()

_ID_EXACT_RE = re.compile(r'^[A-Z]+\d+-\d+$')  # 例：NAD104-3, ACS1-1, P23-2

def _to_base_id(id_or_full: str) -> str:
    s = _norm_id(id_or_full)
    return s.split('-')[0]

def load_dataset_for_model(
    data_root: str,
    embedding_model: str,
    feature_type: str = "difference",
    allowed_ids: Optional[Dict[str, Set[str]]] = None,
    demo_lookup: Optional[Dict[str, Dict[str, float]]] = None, 
    include_demo: bool = True,
    demo_weight: float = 1.0,
    apply_correlation_filter: bool = False,
    correlation_threshold: float = 0.9,
    use_all_visits: bool = False  # 新增參數：是否使用所有次數的資料
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    載入資料集並提取特徵
    use_all_visits: True=使用所有次數的資料, False=只用最新一次
    注意：allowed_ids 中儲存的是 base_id（如 P1, ACS1）
    """
    data_path = Path(data_root)
    X_list, y_list, case_names = [], [], []
    subject_ids = []
    demo_list: List[Tuple[float, float]] = []

    def _maybe_push_demo(case_id: str):
        if not include_demo or demo_lookup is None: 
            return
        # 對於多次收案，提取基礎ID
        base_id, _ = parse_case_id(case_id)
        meta = demo_lookup.get(base_id) or demo_lookup.get(case_id)
        if meta is None:
            demo_list.append((np.nan, np.nan)); return
        age = meta["Age"]
        sex = meta["Sex"]
        if pd.isna(sex):
            sex = demo_lookup.get("_SEX_MODE_", np.nan)
        demo_list.append((age, sex))

    # ---- 在 _maybe_push_demo 之後加上 ----
    allowed_bases_map: Dict[str, Set[str]] = {}
    if allowed_ids:
        for g in ["ACS", "NAD", "P"]:
            if g in allowed_ids and allowed_ids[g]:
                # 不論給 base 或 full-id，一律轉成 base_id 做 subject-level 篩選
                allowed_bases_map[g] = {_to_base_id(x) for x in allowed_ids[g]}

    def _filter_by_base(case_folders_to_use: List[Path], group_key: str) -> List[Path]:
        if not allowed_ids or group_key not in allowed_bases_map:
            return case_folders_to_use
        allow_bases = allowed_bases_map[group_key]
        kept = []
        for folder in case_folders_to_use:
            base_id, _ = parse_case_id(folder.name)
            if _to_base_id(base_id) in allow_bases:
                kept.append(folder)
        return kept

    
    # --- health 類別
    health_path = data_path / "health"
    if health_path.exists():
        print(f"  處理 health 類別 ({embedding_model}, {feature_type}, {'所有次數' if use_all_visits else '最新一次'})...")
        for category_folder in health_path.iterdir():
            if category_folder.is_dir():
                group_name = category_folder.name
                all_case_folders = [f for f in category_folder.iterdir() if f.is_dir()]
                
                # 選擇使用所有次數或只用最新一次
                if use_all_visits:
                    case_folders_to_use = get_all_cases(all_case_folders)
                else:
                    case_folders_to_use = filter_latest_cases(all_case_folders)

                # 根據 allowed_ids（受試者層級：base_id）進行篩選
                before = len(case_folders_to_use)
                case_folders_to_use = _filter_by_base(case_folders_to_use, group_name)
                if allowed_ids and group_name in allowed_bases_map:
                    print(f"    {category_folder.name}: {len(case_folders_to_use)} 個樣本（原 {before}，篩選後；依 base_id）")
                else:
                    print(f"    {category_folder.name}: {len(case_folders_to_use)} 個樣本（無篩選）")

                for case_folder in case_folders_to_use:
                    features = extract_features_from_case(case_folder, embedding_model, feature_type)
                    if features is not None:
                        X_list.append(features)
                        y_list.append(0)
                        case_names.append(f"health/{category_folder.name}/{case_folder.name}")
                        # 提取基礎ID作為subject ID
                        base_id, _ = parse_case_id(case_folder.name)
                        subject_ids.append(base_id)
                        _maybe_push_demo(case_folder.name)

    # --- patient 類別
    patient_path = data_path / "patient"
    if patient_path.exists():
        print(f"  處理 patient 類別 ({embedding_model}, {feature_type}, {'所有次數' if use_all_visits else '最新一次'})...")
        all_case_folders = [f for f in patient_path.iterdir() if f.is_dir()]
        
        # 選擇使用所有次數或只用最新一次
        if use_all_visits:
            case_folders_to_use = get_all_cases(all_case_folders)
        else:
            case_folders_to_use = filter_latest_cases(all_case_folders)
        
        # 根據allowed_ids進行篩選（allowed_ids儲存的是base_id）
        before = len(case_folders_to_use)
        case_folders_to_use = _filter_by_base(case_folders_to_use, "P")
        if allowed_ids and "P" in allowed_bases_map:
            print(f"    找到 {len(case_folders_to_use)} 個樣本（原 {before}，篩選後；依 base_id）")
        else:
            print(f"    找到 {len(case_folders_to_use)} 個樣本（無篩選）")

        for case_folder in case_folders_to_use:
            features = extract_features_from_case(case_folder, embedding_model, feature_type)
            if features is not None:
                X_list.append(features)
                y_list.append(1)
                case_names.append(f"patient/{case_folder.name}")
                # 提取基礎ID作為subject ID
                base_id, _ = parse_case_id(case_folder.name)
                subject_ids.append(base_id)
                _maybe_push_demo(case_folder.name)

    # --- 尺寸對齊
    if X_list:
        if feature_type != "relative":
            max_len = max(len(x) for x in X_list)
            X_list_padded = []
            for x in X_list:
                if len(x) < max_len:
                    X_list_padded.append(np.pad(x, (0, max_len - len(x)), 'constant', constant_values=0))
                elif len(x) > max_len:
                    X_list_padded.append(x[:max_len])
                else:
                    X_list_padded.append(x)
            X_embed = np.asarray(X_list_padded, dtype=np.float64)
        else:
            X_embed = np.asarray(X_list, dtype=np.float64)
            if len(X_embed.shape) == 1:
                X_embed = X_embed.reshape(-1, 1)
    else:
        return np.array([]), np.array([]), [], []

    y = np.asarray(y_list, dtype=int)

    # --- 應用相關係數過濾
    if apply_correlation_filter and feature_type != "relative" and X_embed.shape[1] > 1:
        print(f"  應用相關係數過濾 (閾值={correlation_threshold})...")
        X_embed, kept_features = remove_highly_correlated_features(X_embed, correlation_threshold)

    # --- 標準化與串接 demo
    if include_demo and demo_lookup is not None:
        demo_arr = np.array(demo_list, dtype=np.float64)
        # 補值處理
        if np.isnan(demo_arr[:,0]).any():
            age_mean = np.nanmean(demo_arr[:,0])
            demo_arr[np.isnan(demo_arr[:,0]), 0] = age_mean
        if np.isnan(demo_arr[:,1]).any():
            sex_mode = demo_lookup.get("_SEX_MODE_", np.nan)
            fill_val = sex_mode if not np.isnan(sex_mode) else 0.5
            demo_arr[np.isnan(demo_arr[:,1]), 1] = fill_val

        # 標準化
        embed_mean = X_embed.mean(axis=0, keepdims=True)
        embed_std  = X_embed.std(axis=0, keepdims=True)
        embed_std[embed_std == 0] = 1.0
        X_embed_z = (X_embed - embed_mean) / embed_std

        demo_mean = demo_arr.mean(axis=0, keepdims=True)
        demo_std  = demo_arr.std(axis=0, keepdims=True)
        demo_std[demo_std == 0] = 1.0
        demo_z = (demo_arr - demo_mean) / demo_std

        X = np.hstack([X_embed_z, demo_weight * demo_z])
    else:
        X = X_embed

    return X, y, case_names, subject_ids

def leave_one_subject_out_cv(
    X: np.ndarray, 
    y: np.ndarray, 
    subject_ids: List[str],
    classifier_name: str,
    use_xgb_feature_selection: bool = False,
    xgb_importance_ratio: float = 0.8
) -> Dict:
    """
    Leave-One-Subject-Out 交叉驗證
    """
    unique_subjects = list(set(subject_ids))
    n_subjects = len(unique_subjects)
    
    print(f"    執行 LOSO 交叉驗證，共 {n_subjects} 個獨立受試者")
    
    all_y_true = []
    all_y_pred = []
    all_test_subjects = []
    
    for i, test_subject in enumerate(unique_subjects):
        # 建立訓練和測試索引
        test_indices = [idx for idx, sid in enumerate(subject_ids) if sid == test_subject]
        train_indices = [idx for idx, sid in enumerate(subject_ids) if sid != test_subject]
        
        if not test_indices or not train_indices:
            continue
            
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # XGBoost 特徵篩選（如果啟用且為XGBoost分類器）
        if classifier_name == 'XGBoost' and use_xgb_feature_selection:
            X_train, X_test, _ = select_features_by_xgb_importance(
                X_train, y_train, X_test, xgb_importance_ratio
            )
        
        # 標準化
        use_scaler = classifier_name in ('SVM', 'Logistic Regression')
        
        if use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # 訓練模型
        if classifier_name == 'Random Forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        elif classifier_name == 'SVM':
            classifier = SVC(kernel='rbf', random_state=42)
        elif classifier_name == 'Logistic Regression':
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier_name == 'XGBoost':
            classifier = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=5, n_jobs=-1)
        
        classifier.fit(X_train_scaled, y_train)
        
        # 預測
        y_pred = classifier.predict(X_test_scaled)
        
        # 如果該受試者有多個樣本，取多數決
        if len(y_pred) > 1:
            y_pred_final = 1 if np.mean(y_pred) >= 0.5 else 0
            y_true_final = 1 if np.mean(y_test) >= 0.5 else 0
        else:
            y_pred_final = y_pred[0]
            y_true_final = y_test[0]
        
        all_y_true.append(y_true_final)
        all_y_pred.append(y_pred_final)
        all_test_subjects.append(test_subject)
        
        if (i + 1) % 20 == 0:
            print(f"      進度: {i + 1}/{n_subjects}")
    
    # 計算整體指標
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    mcc = matthews_corrcoef(all_y_true, all_y_pred)
    acc = accuracy_score(all_y_true, all_y_pred)
    
    return {
        'confusion_matrix': cm,
        'mcc': mcc,
        'accuracy': acc,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'test_subjects': all_test_subjects
    }

def k_fold_cv(
    X: np.ndarray, 
    y: np.ndarray, 
    subject_ids: List[str],
    classifier_name: str,
    n_folds: int = 5,
    use_xgb_feature_selection: bool = False,
    xgb_importance_ratio: float = 0.8
) -> Dict:
    """
    K-Fold 交叉驗證（考慮受試者分組）
    """
    print(f"    執行 {n_folds}-fold 交叉驗證")
    
    # 建立受試者到索引的映射
    unique_subjects = list(set(subject_ids))
    subject_to_indices = {subj: [] for subj in unique_subjects}
    for idx, subj in enumerate(subject_ids):
        subject_to_indices[subj].append(idx)
    
    # 建立受試者級別的標籤
    subject_labels = []
    for subj in unique_subjects:
        indices = subject_to_indices[subj]
        # 取該受試者所有樣本標籤的眾數
        labels = [y[i] for i in indices]
        subject_label = 1 if np.mean(labels) >= 0.5 else 0
        subject_labels.append(subject_label)
    
    subject_labels = np.array(subject_labels)
    
    # 使用StratifiedKFold對受試者進行分組
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    
    for fold_idx, (train_subject_idx, test_subject_idx) in enumerate(skf.split(unique_subjects, subject_labels)):
        print(f"      Fold {fold_idx + 1}/{n_folds}")
        
        # 獲取訓練和測試受試者
        train_subjects = [unique_subjects[i] for i in train_subject_idx]
        test_subjects = [unique_subjects[i] for i in test_subject_idx]
        
        # 獲取對應的樣本索引
        train_indices = []
        test_indices = []
        for subj in train_subjects:
            train_indices.extend(subject_to_indices[subj])
        for subj in test_subjects:
            test_indices.extend(subject_to_indices[subj])
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # XGBoost 特徵篩選
        if classifier_name == 'XGBoost' and use_xgb_feature_selection:
            X_train, X_test, _ = select_features_by_xgb_importance(
                X_train, y_train, X_test, xgb_importance_ratio
            )
        
        # 標準化
        use_scaler = classifier_name in ('SVM', 'Logistic Regression')
        
        if use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # 訓練模型
        if classifier_name == 'Random Forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        elif classifier_name == 'SVM':
            classifier = SVC(kernel='rbf', random_state=42)
        elif classifier_name == 'Logistic Regression':
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier_name == 'XGBoost':
            classifier = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=5, n_jobs=-1)
        
        classifier.fit(X_train_scaled, y_train)
        
        # 預測（對每個測試受試者進行預測）
        for test_subj in test_subjects:
            subj_indices = [i for i in test_indices if subject_ids[i] == test_subj]
            subj_test_idx = [test_indices.index(i) for i in subj_indices]
            
            y_pred_subj = classifier.predict(X_test_scaled[subj_test_idx])
            y_true_subj = y_test[subj_test_idx]
            
            # 取多數決
            if len(y_pred_subj) > 1:
                y_pred_final = 1 if np.mean(y_pred_subj) >= 0.5 else 0
                y_true_final = 1 if np.mean(y_true_subj) >= 0.5 else 0
            else:
                y_pred_final = y_pred_subj[0]
                y_true_final = y_true_subj[0]
            
            all_y_true.append(y_true_final)
            all_y_pred.append(y_pred_final)
    
    # 計算整體指標
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    mcc = matthews_corrcoef(all_y_true, all_y_pred)
    acc = accuracy_score(all_y_true, all_y_pred)
    
    return {
        'confusion_matrix': cm,
        'mcc': mcc,
        'accuracy': acc,
        'y_true': all_y_true,
        'y_pred': all_y_pred
    }

def train_with_cv(
    X: np.ndarray, 
    y: np.ndarray, 
    subject_ids: List[str],
    embedding_model: str, 
    feature_type: str,
    cv_method: str = "LOSO",  # "LOSO" or "5-fold"
    use_xgb_feature_selection: bool = False,
    xgb_importance_ratio: float = 0.8,
    classifier_types: Optional[List[str]] = None,
    output_dir: str = "model_output"
) -> Dict:
    """
    使用交叉驗證訓練多個分類器
    cv_method: "LOSO" 或 "5-fold"
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for classifier_type in classifier_types:
        print(f"\n  訓練 {classifier_type} ({embedding_model}, {feature_type}, {cv_method})...")
        
        if cv_method == "LOSO":
            result = leave_one_subject_out_cv(
                X, y, subject_ids, classifier_type,
                use_xgb_feature_selection=use_xgb_feature_selection,
                xgb_importance_ratio=xgb_importance_ratio
            )
        elif cv_method == "5-fold":
            result = k_fold_cv(
                X, y, subject_ids, classifier_type,
                n_folds=5,
                use_xgb_feature_selection=use_xgb_feature_selection,
                xgb_importance_ratio=xgb_importance_ratio
            )
        else:
            raise ValueError(f"不支援的CV方法: {cv_method}")
        
        cm = result['confusion_matrix']
        mcc = result['mcc']
        acc = result['accuracy']
        
        results[classifier_type] = {
            'confusion_matrix': cm.tolist(),
            'mcc': float(mcc),
            'accuracy': float(acc),
            'sensitivity': float(cm[1,1] / (cm[1,0] + cm[1,1])) if (cm[1,0] + cm[1,1]) > 0 else 0,
            'specificity': float(cm[0,0] / (cm[0,0] + cm[0,1])) if (cm[0,0] + cm[0,1]) > 0 else 0,
            'cv_method': cv_method
        }
        
        print(f"    準確率: {acc:.4f}")
        print(f"    MCC: {mcc:.4f}")
        print(f"    靈敏度: {results[classifier_type]['sensitivity']:.4f}")
        print(f"    特異度: {results[classifier_type]['specificity']:.4f}")
        print(f"    混淆矩陣:")
        print(f"      TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"      FN={cm[1,0]}, TP={cm[1,1]}")
    
    return results

def main():
    """主程式"""
    # 基礎路徑設定
    input_root = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature_V2\datung"
    output_root = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\results\DeepLearning\20250923_enhanced"

    # 年齡表來源
    p_csv   = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\p_merged.csv"
    acs_csv = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\ACS_merged_results.csv"
    nad_csv = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\NAD_merged_results.csv"
    excel_file = None
    
    # ============================
    # 控制參數設定（新增）
    # ============================
    ENABLE_AGE_MATCHING = False     # 設為 True 啟用年齡配對，False 停用
    ENABLE_CDR_FILTER = True        # 設為 True 啟用 CDR 篩選，False 停用
    CDR_THRESHOLDS = [0.5, 1, 2] if ENABLE_CDR_FILTER else [None]
    CLASSIFIER_TYPES = ['Random Forest', 'XGBoost', 'SVM', 'Logistic Regression']


    # 新增控制參數
    USE_ALL_VISITS = True           # True: 使用所有次數的資料, False: 只用最新一次
    CV_METHOD = "5-fold"             # "LOSO" 或 "5-fold"
    USE_XGB_FEATURE_SELECTION = True # 是否使用XGBoost特徵篩選
    XGB_IMPORTANCE_RATIO = 0.8      # 保留的特徵重要性比例
    
    print("\n" + "="*60)
    print("執行設定:")
    print(f"  年齡配對: {'啟用' if ENABLE_AGE_MATCHING else '停用'}")
    print(f"  CDR篩選: {'啟用' if ENABLE_CDR_FILTER else '停用'}")
    if ENABLE_CDR_FILTER:
        print(f"  CDR閾值: {CDR_THRESHOLDS}")
    print(f"  資料選擇: {'所有次數' if USE_ALL_VISITS else '只用最新一次'}")
    print(f"  交叉驗證方法: {CV_METHOD}")
    print(f"  XGBoost特徵篩選: {'啟用' if USE_XGB_FEATURE_SELECTION else '停用'}")
    if USE_XGB_FEATURE_SELECTION:
        print(f"  特徵重要性閾值: {XGB_IMPORTANCE_RATIO*100:.0f}%")
    print("="*60)

    # 總體結果儲存
    overall_results = {}

    for cdr_threshold in CDR_THRESHOLDS:
        cdr_results = {}

        if ENABLE_CDR_FILTER:
            print(f"\n{'#'*60}")
            print(f"使用 CDR 閾值 > {cdr_threshold} 進行篩選")
            print(f"{'#'*60}")
        else:
            print(f"\n{'#'*60}")
            print(f"不使用 CDR 篩選")
            print(f"{'#'*60}")
        
        # 載入並配對年齡表
        allowed_ids = None
        demo_lookup = None
        
        try:
            age_tables = load_age_tables(
                p_source=p_csv, 
                acs_source=acs_csv, 
                nad_source=nad_csv, 
                excel_source=excel_file
            )
            
            allowed_ids, age_summary = age_balance_ids_two_groups(
                age_tables, 
                nbins=5, 
                seed=42, 
                method="quantile",
                enable_age_matching=ENABLE_AGE_MATCHING,
                cdr_threshold=cdr_threshold if ENABLE_CDR_FILTER else None
            )
            
            demo_lookup = build_demo_lookup(age_tables)
            
            if ENABLE_AGE_MATCHING:
                print("\n年齡配對完成，並建立 Age/Sex 查表。")
            else:
                print("\n跳過年齡配對，僅建立 Age/Sex 查表。")
                
        except Exception as e:
            print(f"\n[警告] 處理失敗：{e}")
            continue
        
        # 設定
        embedding_models = ['vggface', 'arcface', 'dlib', 'deepid', 'topofr']
        feature_types = ['difference', 'average', 'relative']
        demo_weight = 1.0
        
        # 對每個嵌入模型和特徵類型
        for embedding_model in embedding_models:
            for feature_type in feature_types:
                print(f"\n{'='*50}")
                print(f"處理: {embedding_model} - {feature_type}")
                print('='*50)
                
                apply_filter = (feature_type in ['average', 'relative'] and feature_type != 'relative')
                
                X, y, case_names, subject_ids = load_dataset_for_model(
                    input_root, 
                    embedding_model,
                    feature_type=feature_type,
                    allowed_ids=allowed_ids,
                    demo_lookup=demo_lookup,
                    include_demo=True,
                    demo_weight=demo_weight,
                    apply_correlation_filter=apply_filter,
                    correlation_threshold=0.9,
                    use_all_visits=USE_ALL_VISITS  # 使用新參數
                )
                
                if len(X) == 0:
                    print(f"  警告: {embedding_model}-{feature_type} 沒有載入到任何資料")
                    continue
                
                print(f"\n  最終資料集:")
                print(f"    總樣本數: {len(X)}")
                print(f"    獨立受試者數: {len(set(subject_ids))}")
                print(f"    Health: {np.sum(y == 0)}, Patient: {np.sum(y == 1)}")
                print(f"    特徵維度: {X.shape[1]}")
                
                # 使用新的訓練函數
                results = train_with_cv(
                    X, y, subject_ids,
                    embedding_model, 
                    feature_type,
                    cv_method=CV_METHOD,
                    use_xgb_feature_selection=USE_XGB_FEATURE_SELECTION,
                    xgb_importance_ratio=XGB_IMPORTANCE_RATIO,
                    classifier_types=CLASSIFIER_TYPES,
                    output_dir=output_root
                )
                
                model_key = f"{embedding_model}_{feature_type}"
                cdr_results[model_key] = results

        if ENABLE_CDR_FILTER:
            result_key = f"CDR_gt_{cdr_threshold}"
        else:
            result_key = "No_filtering"
            
        overall_results[result_key] = cdr_results

    # 儲存所有結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 根據設定調整輸出檔名
    suffix_parts = []
    if not ENABLE_AGE_MATCHING:
        suffix_parts.append("no_age")
    if ENABLE_CDR_FILTER:
        suffix_parts.append("cdr_filter")
    if USE_ALL_VISITS:
        suffix_parts.append("all_visits")
    suffix_parts.append(CV_METHOD.replace("-", ""))
    if USE_XGB_FEATURE_SELECTION:
        suffix_parts.append(f"xgb_feat{int(XGB_IMPORTANCE_RATIO*100)}")
    
    suffix = "_".join(suffix_parts) if suffix_parts else "default"
    
    results_filename = Path(output_root) / f"all_cv_results_{suffix}_{timestamp}.json"
    
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("訓練完成！")
    print(f"結果已儲存至 '{results_filename}'")
    
    # 總結報告
    print(f"\n總結報告 ({CV_METHOD}):")
    print("-"*60)
    for cdr_key, models in overall_results.items():
        print(f"\n篩選條件: {cdr_key}")
        for model_key, clf_dict in models.items():
            print(f"  {model_key}:")
            for clf_name, metrics in clf_dict.items():
                acc = metrics.get('accuracy')
                mcc = metrics.get('mcc')
                if acc is None or mcc is None:
                    print(f"    {clf_name}: (缺少指標)")
                    continue
                print(f"    {clf_name}:")
                print(f"      準確率: {acc:.4f}")
                print(f"      MCC: {mcc:.4f}")

if __name__ == "__main__":
    main()