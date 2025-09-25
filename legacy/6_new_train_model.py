# import json
# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple, Optional, Set   # === NEW ===
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, classification_report
# import xgboost as xgb
# import joblib
# from datetime import datetime
# import re
# import warnings
# warnings.filterwarnings('ignore')

# def parse_case_id(folder_name: str) -> Tuple[str, int]:
#     """
#     解析資料夾名稱，提取個案ID和收案次數
#     例如: P1-2 -> (P1, 2), ACS1-1 -> (ACS1, 1)
#     """
#     patterns = [
#         r'^([A-Za-z]+\d+)-(\d+)$',
#         r'^([A-Za-z]+)-(\d+)$',
#         r'^(\w+)-(\d+)$'
#     ]
#     for pattern in patterns:
#         match = re.match(pattern, folder_name)
#         if match:
#             case_id = match.group(1)
#             visit_number = int(match.group(2))
#             return case_id, visit_number
#     return folder_name, 0

# def filter_latest_cases(case_folders: List[Path]) -> List[Path]:
#     """
#     過濾出每個個案的最後一次收案資料
#     """
#     case_dict = {}
#     for folder in case_folders:
#         folder_name = folder.name
#         case_id, visit_number = parse_case_id(folder_name)
#         if case_id not in case_dict:
#             case_dict[case_id] = []
#         case_dict[case_id].append((visit_number, folder))
#     latest_folders = []
#     for case_id, visits in case_dict.items():
#         visits.sort(key=lambda x: x[0], reverse=True)
#         latest_folders.append(visits[0][1])
#         if len(visits) > 1:
#             print(f"    個案 {case_id}: 找到 {len(visits)} 次收案，選擇第 {visits[0][0]} 次")
#     return latest_folders

# def read_difference_json(filepath: str) -> Dict:
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def extract_raw_features_from_case(case_folder: Path, embedding_model: str) -> Optional[np.ndarray]:
#     json_files = sorted(case_folder.glob("*_LR_difference.json"))
#     if not json_files:
#         return None
#     vectors = []
#     for json_file in json_files:
#         data = read_difference_json(json_file)
#         diff = data.get("embedding_differences", {}).get(embedding_model)
#         if diff is not None:
#             v = np.asarray(diff, dtype=float)
#             vectors.append(v)
#     if not vectors:
#         return None
#     shapes = {v.shape for v in vectors}
#     if len(shapes) > 1:
#         raise ValueError(f"向量長度不一致，無法取平均：{sorted(shapes)}")
#     stacked = np.vstack(vectors)
#     mean_vec = stacked.mean(axis=0)
#     return mean_vec

# # ==============================
# # === NEW: 年齡表與配對功能 ===
# # ==============================

# def _select_latest_rows_by_id(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     將同一 base_id（如 P1、ACS3、NAD10）只保留「最後一次收案」那一列。
#     以 ID 字尾 -n 解析，優先用 visit_number 判斷最新；若同分可再看 Photo_Session。
#     """
#     df = df.copy()
#     df["base_id"] = df["ID"].apply(lambda s: parse_case_id(str(s))[0])
#     df["visit_number"] = df["ID"].apply(lambda s: parse_case_id(str(s))[1])

#     def pick_last(group: pd.DataFrame) -> pd.Series:
#         g = group.sort_values(
#             by=["visit_number", "Photo_Session"] if "Photo_Session" in group.columns else ["visit_number"],
#             ascending=False
#         )
#         return g.iloc[0]

#     latest = df.groupby("base_id", as_index=False, group_keys=False).apply(pick_last)
#     latest = latest.drop(columns=["base_id", "visit_number"])
#     return latest

# def _read_age_table_generic(path: str) -> pd.DataFrame:
#     """
#     讀單一檔（csv/xlsx），至少需要欄位：ID, Age。會自動：
#       - 嘗試多種編碼（含 utf-8-sig 以去除 BOM）
#       - 正規化欄名（去空白/零寬字元、轉小寫）
#       - 映射常見同義欄名到 ID / Age
#     讀不到時會回報實際欄名與正規化欄名，便於除錯。
#     """
#     p = Path(path)

#     # 1) 讀檔
#     if p.suffix.lower() in [".xlsx", ".xls"]:
#         df = pd.read_excel(p)
#     else:
#         # 逐一嘗試常見編碼
#         tried_encs = []
#         df = None
#         for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
#             try:
#                 df = pd.read_csv(p, sep=None, engine="python", encoding=enc)
#                 break
#             except Exception as e:
#                 tried_encs.append(f"{enc}: {e}")
#         if df is None:
#             raise RuntimeError(f"讀取 CSV 失敗（嘗試編碼：{'; '.join(tried_encs)}）")

#     # 2) 正規化欄名
#     def _normalize_col(s: str) -> str:
#         s = str(s)
#         # 去 BOM、零寬字元
#         s = s.replace("\ufeff", "").replace("\u200b", "")
#         # 全形括號→半形
#         s = s.replace("（", "(").replace("）", ")")
#         # 去前後空白
#         s = s.strip()
#         # 去內部所有空白
#         s = re.sub(r"\s+", "", s)
#         return s.lower()

#     raw_cols = list(df.columns)
#     norm_cols = {_normalize_col(c): c for c in df.columns}

#     # 3) 欄名映射 → 統一成 'ID' 與 'Age'
#     id_keys  = {"id", "caseid", "subjectid", "pid", "個案id", "編號"}
#     age_keys = {"age", "年齡", "ageyears", "age_years"}
#     sex_keys = {"sex", "gender", "性別"}

#     rename_map = {}
#     found_id_src = found_age_src = found_sex_src = None

#     for norm, src in norm_cols.items():
#         if norm in id_keys and found_id_src is None:
#             rename_map[src] = "ID"; found_id_src = src
#         if norm in age_keys and found_age_src is None:
#             rename_map[src] = "Age"; found_age_src = src
#         if norm in sex_keys and found_sex_src is None:
#             rename_map[src] = "Sex"; found_sex_src = src

#     df = df.rename(columns=rename_map)

#     # 4) 最終檢查 + 詳細錯誤訊息
#     if "ID" not in df.columns or "Age" not in df.columns:
#         debug_msg = (
#             f"{path} 缺少必要欄位：ID 或 Age\n"
#             f"實際讀到欄位：{raw_cols}\n"
#             f"正規化後欄位：{[_normalize_col(c) for c in raw_cols]}\n"
#             f"（提示：若第一欄是 \\ufeffID，請確認使用 utf-8-sig；或欄名是否有空白/全形/零寬字元）"
#         )
#         raise ValueError(debug_msg)

#     # 5) 清理資料
#     df = df.dropna(subset=["ID", "Age"])
#     df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
#     df = df.dropna(subset=["Age"])
#     if "Sex" in df.columns:
#         def _parse_sex(x):
#             if pd.isna(x): return np.nan
#             s = str(x).strip().lower()
#             if s in {"m", "male", "man", "boy", "1", "男", "男性"}: return 1
#             if s in {"f", "female", "woman", "girl", "0", "女", "女性"}: return 0
#             return np.nan
#         df["Sex"] = df["Sex"].apply(_parse_sex)

#     return df

# def load_age_tables(
#     p_source: Optional[str] = None,
#     acs_source: Optional[str] = None,
#     nad_source: Optional[str] = None,
#     excel_source: Optional[str] = None
# ) -> Dict[str, pd.DataFrame]:
#     """
#     讀入 P/ACS/NAD 年齡表。
#     支援：
#       1) 三個獨立檔案 (csv/xlsx)：p_source, acs_source, nad_source
#       2) 一個 Excel：excel_source（包含三個 sheet，名稱含 'P'、'ACS'、'NAD' 的任一）
#     回傳 dict: {'P': df_p, 'ACS': df_acs, 'NAD': df_nad}，每個 df 為「最後一次收案」版本。
#     """
#     if excel_source:
#         xl = pd.ExcelFile(excel_source)
#         sheets = {name.lower(): name for name in xl.sheet_names}
#         # 嘗試用名稱關鍵字找表
#         def pick_sheet(keyword: str) -> str:
#             for k, v in sheets.items():
#                 if keyword.lower() in k:
#                     return v
#             raise ValueError(f"Excel 中找不到包含 '{keyword}' 的工作表")
#         p_df = pd.read_excel(excel_source, sheet_name=pick_sheet("P"))
#         acs_df = pd.read_excel(excel_source, sheet_name=pick_sheet("ACS"))
#         nad_df = pd.read_excel(excel_source, sheet_name=pick_sheet("NAD"))
#     else:
#         if not (p_source and acs_source and nad_source):
#             raise ValueError("請提供 excel_source，或同時提供 p_source、acs_source、nad_source 三個路徑")
#         p_df = _read_age_table_generic(p_source)
#         acs_df = _read_age_table_generic(acs_source)
#         nad_df = _read_age_table_generic(nad_source)

#     # 只留最後一次收案
#     p_df = _select_latest_rows_by_id(p_df)
#     acs_df = _select_latest_rows_by_id(acs_df)
#     nad_df = _select_latest_rows_by_id(nad_df)

#     return {"P": p_df, "ACS": acs_df, "NAD": nad_df}

# def age_balance_ids(
#     tables: Dict[str, pd.DataFrame],
#     nbins: int = 5,
#     seed: int = 42,
#     method: str = "quantile"
# ) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
#     """
#     以分箱對齊方式使三群年齡分布更接近，回傳每群被選中的 ID 集合，以及選後的統計摘要。
#     - method='quantile'：用整體年齡分布做 nbins 個分位數分箱（較穩健）
#     - method='fixed'：用固定箱寬（例如 5 歲；這個模式可按需求再擴充）
#     """
#     rng = np.random.RandomState(seed)

#     # 準備合併資料（只取 ID, Age）
#     df_all = []
#     for g in ["ACS", "NAD", "P"]:
#         df = tables[g][["ID", "Age"]].copy()
#         df["group"] = g
#         df_all.append(df)
#     all_df = pd.concat(df_all, ignore_index=True)

#     # 建立分箱
#     ages = all_df["Age"].values
#     if method == "quantile":
#         # 以全體年齡做分位數箱
#         try:
#             all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins, duplicates="drop")
#         except ValueError:
#             # 資料太少或年齡重複，退回較少箱數
#             nbins_eff = max(2, min(nbins, all_df["Age"].nunique()))
#             all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins_eff, duplicates="drop")
#     else:
#         # 你若想用固定箱寬，可改為 pd.cut(…, bins=np.arange(min_age, max_age+5, 5))
#         raise NotImplementedError("目前僅實作 method='quantile'")

#     # 逐箱對齊抽樣
#     selected_ids: Dict[str, Set[str]] = {"ACS": set(), "NAD": set(), "P": set()}
#     bin_levels = all_df["age_bin"].cat.categories

#     for b in bin_levels:
#         bin_df = all_df[all_df["age_bin"] == b]
#         counts = bin_df.groupby("group")["ID"].count().reindex(["ACS", "NAD", "P"]).fillna(0).astype(int)
#         target = counts.min()  # 三群共同可用的最小數量
#         if target == 0:
#             continue
#         # 在該箱內，對每群隨機抽 target 個（不重複）
#         for g in ["ACS", "NAD", "P"]:
#             g_pool = bin_df[bin_df["group"] == g]
#             if len(g_pool) >= target:
#                 pick = g_pool.sample(n=target, random_state=rng)
#                 selected_ids[g].update(pick["ID"].tolist())

#     # 生成選後統計摘要
#     stats = []
#     for g in ["ACS", "NAD", "P"]:
#         sub = all_df[all_df["ID"].isin(selected_ids[g])]
#         stats.append({
#             "group": g,
#             "n_selected": int(sub.shape[0]),
#             "age_mean": float(sub["Age"].mean()) if not sub.empty else np.nan,
#             "age_std": float(sub["Age"].std(ddof=1)) if sub.shape[0] > 1 else np.nan
#         })
#     summary = pd.DataFrame(stats)

#     print("\n[年齡配對後的統計]")
#     for _, row in summary.iterrows():
#         print(f"  {row['group']}: n={row['n_selected']}, mean={row['age_mean']:.2f} , std={row['age_std']:.2f}" if pd.notnull(row['age_mean']) else f"  {row['group']}: n={row['n_selected']}")

#     return selected_ids, summary

# def age_balance_ids_two_groups(
#     tables: Dict[str, pd.DataFrame],
#     nbins: int = 5,
#     seed: int = 42,
#     method: str = "quantile"
# ) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
#     """
#     以兩群（Health=ACS+NAD 合併、P）做年齡分布對齊抽樣。
#     回傳：
#       - allowed_ids: {'ACS': set(...), 'NAD': set(...), 'P': set(...)}
#       - summary: DataFrame，統計 Health 與 P 的 n / mean / std
#     """
#     rng = np.random.RandomState(seed)

#     # --- 準備 Health 與 P 的表 ---
#     acs_df = tables["ACS"][["ID", "Age"]].copy()
#     acs_df["origin"] = "ACS"
#     nad_df = tables["NAD"][["ID", "Age"]].copy()
#     nad_df["origin"] = "NAD"
#     p_df   = tables["P"][["ID", "Age"]].copy()

#     health_df = pd.concat([acs_df, nad_df], ignore_index=True)
#     health_df["group2"] = "Health"
#     p_df["group2"] = "P"

#     all_df = pd.concat([health_df[["ID","Age","group2","origin"]],
#                         p_df[["ID","Age","group2"]]], ignore_index=True)

#     # --- 分箱 ---
#     if method == "quantile":
#         try:
#             all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins, duplicates="drop")
#         except ValueError:
#             nbins_eff = max(2, min(nbins, all_df["Age"].nunique()))
#             all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins_eff, duplicates="drop")
#     else:
#         raise NotImplementedError("目前僅支援 method='quantile'")

#     # --- 每個年齡箱內對齊抽樣（Health vs P）---
#     selected_health_ids: Set[str] = set()
#     selected_p_ids: Set[str] = set()

#     for b in all_df["age_bin"].cat.categories:
#         bin_df = all_df[all_df["age_bin"] == b]
#         n_health = bin_df[bin_df["group2"] == "Health"].shape[0]
#         n_p      = bin_df[bin_df["group2"] == "P"].shape[0]
#         target = min(n_health, n_p)
#         if target == 0:
#             continue

#         # 在該箱內分別抽樣
#         health_pool = bin_df[bin_df["group2"] == "Health"]
#         p_pool      = bin_df[bin_df["group2"] == "P"]

#         pick_h = health_pool.sample(n=target, random_state=rng) if len(health_pool) >= target else health_pool
#         pick_p = p_pool.sample(n=target, random_state=rng) if len(p_pool) >= target else p_pool

#         selected_health_ids.update(pick_h["ID"].tolist())
#         selected_p_ids.update(pick_p["ID"].tolist())

#     # --- Health 再拆回 ACS / NAD ---
#     acs_ids_all = set(tables["ACS"]["ID"].tolist())
#     nad_ids_all = set(tables["NAD"]["ID"].tolist())

#     selected_acs = set([i for i in selected_health_ids if i in acs_ids_all])
#     selected_nad = set([i for i in selected_health_ids if i in nad_ids_all])

#     allowed_ids = {
#         "ACS": selected_acs,
#         "NAD": selected_nad,
#         "P":   selected_p_ids
#     }

#     # --- 生成 Health 與 P 的統計摘要 ---
#     stats = []
#     for g_name, ids in [("Health", selected_health_ids), ("P", selected_p_ids)]:
#         g_sub = all_df[all_df["ID"].isin(ids)]
#         stats.append({
#             "group": g_name,
#             "n_selected": int(g_sub.shape[0]),
#             "age_mean": float(g_sub["Age"].mean()) if not g_sub.empty else np.nan,
#             "age_std": float(g_sub["Age"].std(ddof=1)) if g_sub.shape[0] > 1 else np.nan
#         })
#     summary = pd.DataFrame(stats)

#     print("\n[年齡配對後（Health vs P）的統計]")
#     for _, row in summary.iterrows():
#         if pd.notnull(row["age_mean"]):
#             print(f"  {row['group']}: n={row['n_selected']}, mean={row['age_mean']:.2f}, std={row['age_std']:.2f}")
#         else:
#             print(f"  {row['group']}: n={row['n_selected']}")

#     # 也可順帶印出 Health 內 ACS / NAD 的數量參考
#     print(f"  Health 組內：ACS={len(selected_acs)}, NAD={len(selected_nad)}")

#     return allowed_ids, summary

# def build_demo_lookup(tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
#     """
#     將三表合併為一個查詢：{ID: {'Age': float, 'Sex': 0/1 or np.nan}}
#     （各表已是最後一次收案版本）
#     """
#     frames = []
#     for g in ["P", "ACS", "NAD"]:
#         df = tables[g][["ID", "Age"]].copy()
#         if "Sex" in tables[g].columns:
#             df["Sex"] = tables[g]["Sex"]
#         frames.append(df)
#     merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ID"], keep="first")

#     lookup = {}
#     for _, row in merged.iterrows():
#         lookup[str(row["ID"])] = {
#             "Age": float(row["Age"]),
#             "Sex": (float(row["Sex"]) if "Sex" in merged.columns and pd.notna(row["Sex"]) else np.nan)
#         }
#     # 全域性別眾數（補值用）
#     if "Sex" in merged.columns:
#         sex_mode = merged["Sex"].dropna().mode()
#         lookup["_SEX_MODE_"] = float(sex_mode.iloc[0]) if len(sex_mode)>0 else np.nan
#     else:
#         lookup["_SEX_MODE_"] = np.nan
#     return lookup

# def load_dataset_for_model(
#     data_root: str,
#     embedding_model: str,
#     allowed_ids: Optional[Dict[str, Set[str]]] = None,
#     demo_lookup: Optional[Dict[str, Dict[str, float]]] = None, 
#     include_demo: bool = True,
#     demo_weight: float = 1.0
# ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#     data_path = Path(data_root)
#     X_list, y_list, case_names = [], [], []
#     demo_list: List[Tuple[float, float]] = []

#     def _maybe_push_demo(case_id: str):
#         if not include_demo or demo_lookup is None: 
#             return
#         meta = demo_lookup.get(case_id)
#         if meta is None:
#             # 找不到就記 None（等下再補）
#             demo_list.append((np.nan, np.nan)); return
#         age = meta["Age"]
#         sex = meta["Sex"]
#         if pd.isna(sex):
#             sex = demo_lookup.get("_SEX_MODE_", np.nan)
#         demo_list.append((age, sex))

#     # --- health 類別
#     health_path = data_path / "health"
#     if health_path.exists():
#         print(f"  處理 health 類別 ({embedding_model})...")
#         for category_folder in health_path.iterdir():
#             if category_folder.is_dir():
#                 group_name = category_folder.name  # 'ACS' or 'NAD'
#                 all_case_folders = [f for f in category_folder.iterdir() if f.is_dir()]
#                 latest_case_folders = filter_latest_cases(all_case_folders)
#                 if allowed_ids and group_name in allowed_ids:
#                     before = len(latest_case_folders)
#                     latest_case_folders = [f for f in latest_case_folders if f.name in allowed_ids[group_name]]
#                     print(f"    {category_folder.name}: {len(latest_case_folders)} 個個案（原 {before}，年齡篩選後）")
#                 else:
#                     print(f"    {category_folder.name}: {len(latest_case_folders)} 個個案（無年齡篩選）")

#                 for case_folder in latest_case_folders:
#                     features = extract_raw_features_from_case(case_folder, embedding_model)
#                     if features is not None:
#                         X_list.append(features)
#                         y_list.append(0)
#                         case_names.append(f"health/{category_folder.name}/{case_folder.name}")
#                         _maybe_push_demo(case_folder.name)

#     # --- patient 類別
#     patient_path = data_path / "patient"
#     if patient_path.exists():
#         print(f"  處理 patient 類別 ({embedding_model})...")
#         all_case_folders = [f for f in patient_path.iterdir() if f.is_dir()]
#         latest_case_folders = filter_latest_cases(all_case_folders)
#         if allowed_ids and "P" in allowed_ids:
#             before = len(latest_case_folders)
#             latest_case_folders = [f for f in latest_case_folders if f.name in allowed_ids["P"]]
#             print(f"    找到 {len(latest_case_folders)} 個個案（原 {before}，年齡篩選後）")
#         else:
#             print(f"    找到 {len(latest_case_folders)} 個個案（無年齡篩選）")

#         for case_folder in latest_case_folders:
#             features = extract_raw_features_from_case(case_folder, embedding_model)
#             if features is not None:
#                 X_list.append(features)
#                 y_list.append(1)
#                 case_names.append(f"patient/{case_folder.name}")
#                 _maybe_push_demo(case_folder.name)

#     # --- 尺寸對齊（嵌入向量）
#     if X_list:
#         max_len = max(len(x) for x in X_list)
#         X_list_padded = []
#         for x in X_list:
#             if len(x) < max_len:
#                 X_list_padded.append(np.pad(x, (0, max_len - len(x)), 'constant', constant_values=0))
#             elif len(x) > max_len:
#                 X_list_padded.append(x[:max_len])
#             else:
#                 X_list_padded.append(x)
#         X_embed = np.asarray(X_list_padded, dtype=np.float64)
#     else:
#         return np.array([]), np.array([]), []

#     y = np.asarray(y_list, dtype=int)

#     # --- 分塊標準化與串接 demo
#     if include_demo and demo_lookup is not None:
#         # 將缺值以列平均或性別眾數補值（Age 用資料列平均；Sex 用眾數/0.5 退路）
#         demo_arr = np.array(demo_list, dtype=np.float64)
#         # Age 補值
#         if np.isnan(demo_arr[:,0]).any():
#             age_mean = np.nanmean(demo_arr[:,0])
#             demo_arr[np.isnan(demo_arr[:,0]), 0] = age_mean
#         # Sex 補值
#         if np.isnan(demo_arr[:,1]).any():
#             sex_mode = demo_lookup.get("_SEX_MODE_", np.nan)
#             fill_val = sex_mode if not np.isnan(sex_mode) else 0.5  # 沒資料就 0.5
#             demo_arr[np.isnan(demo_arr[:,1]), 1] = fill_val

#         # 1) 嵌入向量 z-score（以訓練集內部統計）
#         embed_mean = X_embed.mean(axis=0, keepdims=True)
#         embed_std  = X_embed.std(axis=0, keepdims=True)
#         embed_std[embed_std == 0] = 1.0
#         X_embed_z = (X_embed - embed_mean) / embed_std

#         # 2) demo z-score
#         demo_mean = demo_arr.mean(axis=0, keepdims=True)
#         demo_std  = demo_arr.std(axis=0, keepdims=True)
#         demo_std[demo_std == 0] = 1.0
#         demo_z = (demo_arr - demo_mean) / demo_std

#         # 3) 權重
#         X = np.hstack([X_embed_z, demo_weight * demo_z])
#     else:
#         X = X_embed

#     return X, y, case_names

# def train_single_iteration(X, y, classifier_name, random_state):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=random_state, stratify=y
#     )

#     # 只有線性/距離型模型需要標準化；樹模型略過
#     use_scaler = classifier_name in ('SVM', 'Logistic Regression')

#     if use_scaler:
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled  = scaler.transform(X_test)
#     else:
#         scaler = None
#         X_train_scaled, X_test_scaled = X_train, X_test

#     if classifier_name == 'Random Forest':
#         classifier = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
#     elif classifier_name == 'SVM':
#         classifier = SVC(kernel='rbf', random_state=random_state, probability=False)
#     elif classifier_name == 'Logistic Regression':
#         classifier = LogisticRegression(max_iter=1000, random_state=random_state)
#     elif classifier_name == 'XGBoost':
#         classifier = xgb.XGBClassifier(n_estimators=100, random_state=random_state, max_depth=5, n_jobs=-1)

#     classifier.fit(X_train_scaled, y_train)
#     y_pred = classifier.predict(X_test_scaled)

#     cm  = confusion_matrix(y_test, y_pred)
#     mcc = matthews_corrcoef(y_test, y_pred)
#     acc = accuracy_score(y_test, y_pred)

#     return {
#         'confusion_matrix': cm,
#         'mcc': mcc,
#         'accuracy': acc,
#         'model': classifier,
#         'scaler': scaler
#     }


# def train_with_multiple_iterations(X, y, embedding_model, n_iterations=100, output_dir="model_output"):
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
#     classifier_names = ['Random Forest', 'XGBoost']

#     results = {}
#     for classifier_name in classifier_names:
#         print(f"\n  訓練 {classifier_name} ({embedding_model})...")
#         all_cms, all_mccs, all_accs = [], [], []
#         best_model, best_scaler, best_mcc = None, None, -1
#         for i in range(n_iterations):
#             if (i + 1) % 20 == 0:
#                 print(f"    進度: {i + 1}/{n_iterations}")
#             result = train_single_iteration(X, y, classifier_name, random_state=i)
#             all_cms.append(result['confusion_matrix'])
#             all_mccs.append(result['mcc'])
#             all_accs.append(result['accuracy'])
#             if result['mcc'] > best_mcc:
#                 best_mcc = result['mcc']
#                 best_model = result['model']
#                 best_scaler = result['scaler']

#         avg_cm = np.mean(all_cms, axis=0)
#         avg_mcc = np.mean(all_mccs)
#         std_mcc = np.std(all_mccs)
#         avg_acc = np.mean(all_accs)
#         std_acc = np.std(all_accs)

#         results[classifier_name] = {
#             'avg_confusion_matrix': avg_cm.tolist(),
#             'avg_mcc': avg_mcc,
#             'std_mcc': std_mcc,
#             'avg_accuracy': avg_acc,
#             'std_accuracy': std_acc,
#             'all_mccs': all_mccs,
#             'all_accuracies': all_accs,
#             'best_mcc': best_mcc
#         }

#         print(f"    平均準確率: {avg_acc:.4f} ± {std_acc:.4f}")
#         print(f"    平均MCC: {avg_mcc:.4f} ± {std_mcc:.4f}")
#         print(f"    最佳MCC: {best_mcc:.4f}")
#         print(f"    平均混淆矩陣:")
#         print(f"      TN={avg_cm[0,0]:.1f}, FP={avg_cm[0,1]:.1f}")
#         print(f"      FN={avg_cm[1,0]:.1f}, TP={avg_cm[1,1]:.1f}")

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         model_filename = output_path / f"{embedding_model}_{classifier_name.replace(' ', '_')}_{timestamp}.joblib"
#         joblib.dump(best_model, model_filename)
#         scaler_filename = output_path / f"{embedding_model}_{classifier_name.replace(' ', '_')}_scaler_{timestamp}.joblib"
#         joblib.dump(best_scaler, scaler_filename)
#     return results

# def main():
#     """主程式"""
#     # === 你的原始資料路徑 ===
#     data_root = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\_pics\5_vector_diffs\datung"
#     output_dir = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\_pics\6_results_4"

#     # === NEW: 年齡表來源（擇一方式填寫） ===
#     p_csv   = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\p_merged.csv"     # 例如：包含欄位 ID, Age, ...
#     acs_csv = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\ACS_merged_results.csv"
#     nad_csv = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\NAD_merged_results.csv"
#     excel_file = None  # 若改用 Excel 多工作表，填路徑並把上面三個設為 None

#     # === NEW: 載入年齡表、做分箱對齊抽樣，得到允許的 ID 清單 ===
#     allowed_ids = None
#     try:
#         age_tables = load_age_tables(p_source=p_csv, acs_source=acs_csv, nad_source=nad_csv, excel_source=excel_file)
#         allowed_ids, age_summary = age_balance_ids_two_groups(age_tables, nbins=5, seed=42, method="quantile")
#         # === NEW: 建立 demo lookup（含 Age、可能的 Sex）===
#         demo_lookup = build_demo_lookup(age_tables)
#         print("\n年齡配對完成，並建立 Age/Sex 查表。載入資料時會附加到特徵。")
#     except Exception as e:
#         print(f"\n[警告] 年齡表載入/配對失敗，將略過年齡/性別特徵：{e}")

#     embedding_models = ['vggface', 'arcface', 'dlib', 'deepid', 'topofr']
#     all_results = {}

#     demo_weight = 1.0   # 可改成 0.5 / 1.0 / 2.0 / 4.0 做小實驗
#     for embedding_model in embedding_models:
#         print(f"\n{'='*60}")
#         print(f"處理嵌入模型: {embedding_model}")
#         print('='*60)

#         X, y, case_names = load_dataset_for_model(
#             data_root, embedding_model,
#             allowed_ids=allowed_ids,
#             demo_lookup=demo_lookup,       # === NEW ===
#             include_demo=True,             # === NEW ===
#             demo_weight=demo_weight        # === NEW ===
#         )

#         if len(X) == 0:
#             print(f"  警告: {embedding_model} 沒有載入到任何資料")
#             continue

#         print(f"\n  最終資料集:")
#         print(f"    總樣本數: {len(X)}")
#         print(f"    Health: {np.sum(y == 0)}, Patient: {np.sum(y == 1)}")
#         print(f"    特徵維度: {X.shape[1]}（含 demo 特徵）")

#         results = train_with_multiple_iterations(
#             X, y, embedding_model,
#             n_iterations=100,
#             output_dir=output_dir
#         )
#         all_results[embedding_model] = results

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_filename = Path(output_dir) / f"all_results_{timestamp}.json"
#     with open(results_filename, 'w', encoding='utf-8') as f:
#         json.dump(all_results, f, indent=2, ensure_ascii=False)

#     print(f"\n{'='*60}")
#     print("訓練完成！")
#     print(f"模型和結果已儲存至 '{output_dir}' 資料夾")

#     print("\n總結報告:")
#     print("-"*60)
#     for embedding_model, model_results in all_results.items():
#         print(f"\n{embedding_model}:")
#         for classifier_name, metrics in model_results.items():
#             print(f"  {classifier_name}:")
#             print(f"    準確率: {metrics['avg_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
#             print(f"    MCC: {metrics['avg_mcc']:.4f} ± {metrics['std_mcc']:.4f}")

# if __name__ == "__main__":
#     main()

#=======V2=======#

# import json
# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple, Optional, Set
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, classification_report
# import xgboost as xgb
# import joblib
# from datetime import datetime
# import re
# import warnings
# warnings.filterwarnings('ignore')

# def parse_case_id(folder_name: str) -> Tuple[str, int]:
#     """
#     解析資料夾名稱，提取個案ID和收案次數
#     例如: P1-2 -> (P1, 2), ACS1-1 -> (ACS1, 1)
#     """
#     patterns = [
#         r'^([A-Za-z]+\d+)-(\d+)$',
#         r'^([A-Za-z]+)-(\d+)$',
#         r'^(\w+)-(\d+)$'
#     ]
#     for pattern in patterns:
#         match = re.match(pattern, folder_name)
#         if match:
#             case_id = match.group(1)
#             visit_number = int(match.group(2))
#             return case_id, visit_number
#     return folder_name, 0

# def filter_latest_cases(case_folders: List[Path]) -> List[Path]:
#     """
#     過濾出每個個案的最後一次收案資料
#     """
#     case_dict = {}
#     for folder in case_folders:
#         folder_name = folder.name
#         case_id, visit_number = parse_case_id(folder_name)
#         if case_id not in case_dict:
#             case_dict[case_id] = []
#         case_dict[case_id].append((visit_number, folder))
#     latest_folders = []
#     for case_id, visits in case_dict.items():
#         visits.sort(key=lambda x: x[0], reverse=True)
#         latest_folders.append(visits[0][1])
#         # if len(visits) > 1:
#         #     print(f"    個案 {case_id}: 找到 {len(visits)} 次收案，選擇第 {visits[0][0]} 次")
#     return latest_folders

# def read_difference_json(filepath: str) -> Dict:
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def extract_features_from_case(case_folder: Path, embedding_model: str, feature_type: str = "difference") -> Optional[np.ndarray]:
#     """
#     從個案資料夾提取指定類型的特徵
#     feature_type: 'difference', 'average', 'relative'
#     """
#     json_files = sorted(case_folder.glob("*_LR_difference.json"))
#     if not json_files:
#         return None
    
#     vectors = []
#     for json_file in json_files:
#         data = read_difference_json(json_file)
        
#         if feature_type == "difference":
#             # 方法0：原始差值特徵
#             feat = data.get("embedding_differences", {}).get(embedding_model)
#         elif feature_type == "average":
#             # 方法1：平均向量特徵
#             feat = data.get("embedding_averages", {}).get(embedding_model)
#         elif feature_type == "relative":
#             # 方法2：相對差異特徵（標量，需要特殊處理）
#             feat = data.get("relative_differences", {}).get(embedding_model)
#             if feat is not None:
#                 # 相對差異是單一數值，不是向量
#                 vectors.append(float(feat))
#                 continue
#         else:
#             raise ValueError(f"Unknown feature_type: {feature_type}")
        
#         if feat is not None:
#             v = np.asarray(feat, dtype=float)
#             vectors.append(v)
    
#     if not vectors:
#         return None
    
#     # 對於相對差異（標量），直接返回平均值
#     if feature_type == "relative":
#         return np.array([np.mean(vectors)])  # 返回一維陣列
    
#     # 對於向量特徵，檢查維度一致性
#     shapes = {v.shape for v in vectors}
#     if len(shapes) > 1:
#         raise ValueError(f"向量長度不一致，無法取平均：{sorted(shapes)}")
#     stacked = np.vstack(vectors)
#     mean_vec = stacked.mean(axis=0)
#     return mean_vec

# def remove_highly_correlated_features(X: np.ndarray, threshold: float = 0.9) -> Tuple[np.ndarray, List[int]]:
#     """
#     移除高度相關的特徵（相關係數 > threshold）
#     返回處理後的特徵矩陣和保留的特徵索引
#     """
#     if X.shape[1] <= 1:
#         return X, list(range(X.shape[1]))
    
#     # 計算相關係數矩陣
#     corr_matrix = np.corrcoef(X.T)
    
#     # 找出要保留的特徵
#     keep_features = []
#     removed_features = set()
    
#     for i in range(corr_matrix.shape[0]):
#         if i in removed_features:
#             continue
#         keep_features.append(i)
#         # 找出與當前特徵高度相關的其他特徵
#         for j in range(i+1, corr_matrix.shape[1]):
#             if abs(corr_matrix[i, j]) > threshold:
#                 removed_features.add(j)
    
#     print(f"    原始特徵數: {X.shape[1]}, 移除 {len(removed_features)} 個高度相關特徵, 保留 {len(keep_features)} 個")
    
#     # 只保留選中的特徵
#     X_filtered = X[:, keep_features]
#     return X_filtered, keep_features

# # ==============================
# # === 年齡表與配對功能（保持不變） ===
# # ==============================

# def _select_latest_rows_by_id(df: pd.DataFrame) -> pd.DataFrame:
#     """將同一 base_id 只保留最後一次收案"""
#     df = df.copy()
#     df["base_id"] = df["ID"].apply(lambda s: parse_case_id(str(s))[0])
#     df["visit_number"] = df["ID"].apply(lambda s: parse_case_id(str(s))[1])

#     def pick_last(group: pd.DataFrame) -> pd.Series:
#         g = group.sort_values(
#             by=["visit_number", "Photo_Session"] if "Photo_Session" in group.columns else ["visit_number"],
#             ascending=False
#         )
#         return g.iloc[0]

#     latest = df.groupby("base_id", as_index=False, group_keys=False).apply(pick_last)
#     latest = latest.drop(columns=["base_id", "visit_number"])
#     return latest

# def _read_age_table_generic(path: str) -> pd.DataFrame:
#     """讀取年齡表（保持原樣）"""
#     p = Path(path)

#     if p.suffix.lower() in [".xlsx", ".xls"]:
#         df = pd.read_excel(p)
#     else:
#         tried_encs = []
#         df = None
#         for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
#             try:
#                 df = pd.read_csv(p, sep=None, engine="python", encoding=enc)
#                 break
#             except Exception as e:
#                 tried_encs.append(f"{enc}: {e}")
#         if df is None:
#             raise RuntimeError(f"讀取 CSV 失敗（嘗試編碼：{'; '.join(tried_encs)}）")

#     def _normalize_col(s: str) -> str:
#         s = str(s)
#         s = s.replace("\ufeff", "").replace("\u200b", "")
#         s = s.replace("（", "(").replace("）", ")")
#         s = s.strip()
#         s = re.sub(r"\s+", "", s)
#         return s.lower()

#     raw_cols = list(df.columns)
#     norm_cols = {_normalize_col(c): c for c in df.columns}

#     id_keys  = {"id", "caseid", "subjectid", "pid", "個案id", "編號"}
#     age_keys = {"age", "年齡", "ageyears", "age_years"}
#     sex_keys = {"sex", "gender", "性別"}

#     rename_map = {}
#     found_id_src = found_age_src = found_sex_src = None

#     for norm, src in norm_cols.items():
#         if norm in id_keys and found_id_src is None:
#             rename_map[src] = "ID"; found_id_src = src
#         if norm in age_keys and found_age_src is None:
#             rename_map[src] = "Age"; found_age_src = src
#         if norm in sex_keys and found_sex_src is None:
#             rename_map[src] = "Sex"; found_sex_src = src

#     df = df.rename(columns=rename_map)

#     if "ID" not in df.columns or "Age" not in df.columns:
#         debug_msg = (
#             f"{path} 缺少必要欄位：ID 或 Age\n"
#             f"實際讀到欄位：{raw_cols}\n"
#             f"正規化後欄位：{[_normalize_col(c) for c in raw_cols]}"
#         )
#         raise ValueError(debug_msg)

#     df = df.dropna(subset=["ID", "Age"])
#     df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
#     df = df.dropna(subset=["Age"])
#     if "Sex" in df.columns:
#         def _parse_sex(x):
#             if pd.isna(x): return np.nan
#             s = str(x).strip().lower()
#             if s in {"m", "male", "man", "boy", "1", "男", "男性"}: return 1
#             if s in {"f", "female", "woman", "girl", "0", "女", "女性"}: return 0
#             return np.nan
#         df["Sex"] = df["Sex"].apply(_parse_sex)

#     return df

# def load_age_tables(
#     p_source: Optional[str] = None,
#     acs_source: Optional[str] = None,
#     nad_source: Optional[str] = None,
#     excel_source: Optional[str] = None
# ) -> Dict[str, pd.DataFrame]:
#     """載入年齡表"""
#     if excel_source:
#         xl = pd.ExcelFile(excel_source)
#         sheets = {name.lower(): name for name in xl.sheet_names}
#         def pick_sheet(keyword: str) -> str:
#             for k, v in sheets.items():
#                 if keyword.lower() in k:
#                     return v
#             raise ValueError(f"Excel 中找不到包含 '{keyword}' 的工作表")
#         p_df = pd.read_excel(excel_source, sheet_name=pick_sheet("P"))
#         acs_df = pd.read_excel(excel_source, sheet_name=pick_sheet("ACS"))
#         nad_df = pd.read_excel(excel_source, sheet_name=pick_sheet("NAD"))
#     else:
#         if not (p_source and acs_source and nad_source):
#             raise ValueError("請提供 excel_source，或同時提供 p_source、acs_source、nad_source 三個路徑")
#         p_df = _read_age_table_generic(p_source)
#         acs_df = _read_age_table_generic(acs_source)
#         nad_df = _read_age_table_generic(nad_source)

#     p_df = _select_latest_rows_by_id(p_df)
#     acs_df = _select_latest_rows_by_id(acs_df)
#     nad_df = _select_latest_rows_by_id(nad_df)

#     return {"P": p_df, "ACS": acs_df, "NAD": nad_df}

# def age_balance_ids_two_groups(
#     tables: Dict[str, pd.DataFrame],
#     nbins: int = 5,
#     seed: int = 42,
#     method: str = "quantile"
# ) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
#     """兩群年齡配對（Health vs P）"""
#     rng = np.random.RandomState(seed)

#     acs_df = tables["ACS"][["ID", "Age"]].copy()
#     acs_df["origin"] = "ACS"
#     nad_df = tables["NAD"][["ID", "Age"]].copy()
#     nad_df["origin"] = "NAD"
#     p_df   = tables["P"][["ID", "Age"]].copy()

#     health_df = pd.concat([acs_df, nad_df], ignore_index=True)
#     health_df["group2"] = "Health"
#     p_df["group2"] = "P"

#     all_df = pd.concat([health_df[["ID","Age","group2","origin"]],
#                         p_df[["ID","Age","group2"]]], ignore_index=True)

#     if method == "quantile":
#         try:
#             all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins, duplicates="drop")
#         except ValueError:
#             nbins_eff = max(2, min(nbins, all_df["Age"].nunique()))
#             all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins_eff, duplicates="drop")
#     else:
#         raise NotImplementedError("目前僅支援 method='quantile'")

#     selected_health_ids: Set[str] = set()
#     selected_p_ids: Set[str] = set()

#     for b in all_df["age_bin"].cat.categories:
#         bin_df = all_df[all_df["age_bin"] == b]
#         n_health = bin_df[bin_df["group2"] == "Health"].shape[0]
#         n_p      = bin_df[bin_df["group2"] == "P"].shape[0]
#         target = min(n_health, n_p)
#         if target == 0:
#             continue

#         health_pool = bin_df[bin_df["group2"] == "Health"]
#         p_pool      = bin_df[bin_df["group2"] == "P"]

#         pick_h = health_pool.sample(n=target, random_state=rng) if len(health_pool) >= target else health_pool
#         pick_p = p_pool.sample(n=target, random_state=rng) if len(p_pool) >= target else p_pool

#         selected_health_ids.update(pick_h["ID"].tolist())
#         selected_p_ids.update(pick_p["ID"].tolist())

#     acs_ids_all = set(tables["ACS"]["ID"].tolist())
#     nad_ids_all = set(tables["NAD"]["ID"].tolist())

#     selected_acs = set([i for i in selected_health_ids if i in acs_ids_all])
#     selected_nad = set([i for i in selected_health_ids if i in nad_ids_all])

#     allowed_ids = {
#         "ACS": selected_acs,
#         "NAD": selected_nad,
#         "P":   selected_p_ids
#     }

#     stats = []
#     for g_name, ids in [("Health", selected_health_ids), ("P", selected_p_ids)]:
#         g_sub = all_df[all_df["ID"].isin(ids)]
#         stats.append({
#             "group": g_name,
#             "n_selected": int(g_sub.shape[0]),
#             "age_mean": float(g_sub["Age"].mean()) if not g_sub.empty else np.nan,
#             "age_std": float(g_sub["Age"].std(ddof=1)) if g_sub.shape[0] > 1 else np.nan
#         })
#     summary = pd.DataFrame(stats)

#     print("\n[年齡配對後（Health vs P）的統計]")
#     for _, row in summary.iterrows():
#         if pd.notnull(row["age_mean"]):
#             print(f"  {row['group']}: n={row['n_selected']}, mean={row['age_mean']:.2f}, std={row['age_std']:.2f}")
#         else:
#             print(f"  {row['group']}: n={row['n_selected']}")

#     print(f"  Health 組內：ACS={len(selected_acs)}, NAD={len(selected_nad)}")

#     return allowed_ids, summary

# def build_demo_lookup(tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
#     """建立人口學資料查詢表"""
#     frames = []
#     for g in ["P", "ACS", "NAD"]:
#         df = tables[g][["ID", "Age"]].copy()
#         if "Sex" in tables[g].columns:
#             df["Sex"] = tables[g]["Sex"]
#         frames.append(df)
#     merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ID"], keep="first")

#     lookup = {}
#     for _, row in merged.iterrows():
#         lookup[str(row["ID"])] = {
#             "Age": float(row["Age"]),
#             "Sex": (float(row["Sex"]) if "Sex" in merged.columns and pd.notna(row["Sex"]) else np.nan)
#         }
#     if "Sex" in merged.columns:
#         sex_mode = merged["Sex"].dropna().mode()
#         lookup["_SEX_MODE_"] = float(sex_mode.iloc[0]) if len(sex_mode)>0 else np.nan
#     else:
#         lookup["_SEX_MODE_"] = np.nan
#     return lookup

# def load_dataset_for_model(
#     data_root: str,
#     embedding_model: str,
#     feature_type: str = "difference",  # 新增：特徵類型選擇
#     allowed_ids: Optional[Dict[str, Set[str]]] = None,
#     demo_lookup: Optional[Dict[str, Dict[str, float]]] = None, 
#     include_demo: bool = True,
#     demo_weight: float = 1.0,
#     apply_correlation_filter: bool = False,  # 新增：是否應用相關係數過濾
#     correlation_threshold: float = 0.9       # 新增：相關係數閾值
# ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:  # 新增返回subject_ids
#     """
#     載入資料集並提取特徵
#     返回：X, y, case_names, subject_ids（用於LOSO）
#     """
#     data_path = Path(data_root)
#     X_list, y_list, case_names = [], [], []
#     subject_ids = []  # 新增：記錄每個樣本的subject ID
#     demo_list: List[Tuple[float, float]] = []

#     def _maybe_push_demo(case_id: str):
#         if not include_demo or demo_lookup is None: 
#             return
#         meta = demo_lookup.get(case_id)
#         if meta is None:
#             demo_list.append((np.nan, np.nan)); return
#         age = meta["Age"]
#         sex = meta["Sex"]
#         if pd.isna(sex):
#             sex = demo_lookup.get("_SEX_MODE_", np.nan)
#         demo_list.append((age, sex))

#     # --- health 類別
#     health_path = data_path / "health"
#     if health_path.exists():
#         print(f"  處理 health 類別 ({embedding_model}, {feature_type})...")
#         for category_folder in health_path.iterdir():
#             if category_folder.is_dir():
#                 group_name = category_folder.name
#                 all_case_folders = [f for f in category_folder.iterdir() if f.is_dir()]
#                 latest_case_folders = filter_latest_cases(all_case_folders)
#                 if allowed_ids and group_name in allowed_ids:
#                     before = len(latest_case_folders)
#                     latest_case_folders = [f for f in latest_case_folders if f.name in allowed_ids[group_name]]
#                     print(f"    {category_folder.name}: {len(latest_case_folders)} 個個案（原 {before}，年齡篩選後）")
#                 else:
#                     print(f"    {category_folder.name}: {len(latest_case_folders)} 個個案（無年齡篩選）")

#                 for case_folder in latest_case_folders:
#                     features = extract_features_from_case(case_folder, embedding_model, feature_type)
#                     if features is not None:
#                         X_list.append(features)
#                         y_list.append(0)
#                         case_names.append(f"health/{category_folder.name}/{case_folder.name}")
#                         subject_ids.append(case_folder.name)  # 記錄subject ID
#                         _maybe_push_demo(case_folder.name)

#     # --- patient 類別
#     patient_path = data_path / "patient"
#     if patient_path.exists():
#         print(f"  處理 patient 類別 ({embedding_model}, {feature_type})...")
#         all_case_folders = [f for f in patient_path.iterdir() if f.is_dir()]
#         latest_case_folders = filter_latest_cases(all_case_folders)
#         if allowed_ids and "P" in allowed_ids:
#             before = len(latest_case_folders)
#             latest_case_folders = [f for f in latest_case_folders if f.name in allowed_ids["P"]]
#             print(f"    找到 {len(latest_case_folders)} 個個案（原 {before}，年齡篩選後）")
#         else:
#             print(f"    找到 {len(latest_case_folders)} 個個案（無年齡篩選）")

#         for case_folder in latest_case_folders:
#             features = extract_features_from_case(case_folder, embedding_model, feature_type)
#             if features is not None:
#                 X_list.append(features)
#                 y_list.append(1)
#                 case_names.append(f"patient/{case_folder.name}")
#                 subject_ids.append(case_folder.name)  # 記錄subject ID
#                 _maybe_push_demo(case_folder.name)

#     # --- 尺寸對齊
#     if X_list:
#         if feature_type != "relative":  # 相對差異是標量，不需要對齊
#             max_len = max(len(x) for x in X_list)
#             X_list_padded = []
#             for x in X_list:
#                 if len(x) < max_len:
#                     X_list_padded.append(np.pad(x, (0, max_len - len(x)), 'constant', constant_values=0))
#                 elif len(x) > max_len:
#                     X_list_padded.append(x[:max_len])
#                 else:
#                     X_list_padded.append(x)
#             X_embed = np.asarray(X_list_padded, dtype=np.float64)
#         else:
#             X_embed = np.asarray(X_list, dtype=np.float64)
#             if len(X_embed.shape) == 1:
#                 X_embed = X_embed.reshape(-1, 1)
#     else:
#         return np.array([]), np.array([]), [], []

#     y = np.asarray(y_list, dtype=int)

#     # --- 應用相關係數過濾（僅對高維特徵）
#     if apply_correlation_filter and feature_type != "relative" and X_embed.shape[1] > 1:
#         print(f"  應用相關係數過濾 (閾值={correlation_threshold})...")
#         X_embed, kept_features = remove_highly_correlated_features(X_embed, correlation_threshold)

#     # --- 標準化與串接 demo
#     if include_demo and demo_lookup is not None:
#         demo_arr = np.array(demo_list, dtype=np.float64)
#         # 補值處理
#         if np.isnan(demo_arr[:,0]).any():
#             age_mean = np.nanmean(demo_arr[:,0])
#             demo_arr[np.isnan(demo_arr[:,0]), 0] = age_mean
#         if np.isnan(demo_arr[:,1]).any():
#             sex_mode = demo_lookup.get("_SEX_MODE_", np.nan)
#             fill_val = sex_mode if not np.isnan(sex_mode) else 0.5
#             demo_arr[np.isnan(demo_arr[:,1]), 1] = fill_val

#         # 標準化
#         embed_mean = X_embed.mean(axis=0, keepdims=True)
#         embed_std  = X_embed.std(axis=0, keepdims=True)
#         embed_std[embed_std == 0] = 1.0
#         X_embed_z = (X_embed - embed_mean) / embed_std

#         demo_mean = demo_arr.mean(axis=0, keepdims=True)
#         demo_std  = demo_arr.std(axis=0, keepdims=True)
#         demo_std[demo_std == 0] = 1.0
#         demo_z = (demo_arr - demo_mean) / demo_std

#         X = np.hstack([X_embed_z, demo_weight * demo_z])
#     else:
#         X = X_embed

#     return X, y, case_names, subject_ids

# def leave_one_subject_out_cv(
#     X: np.ndarray, 
#     y: np.ndarray, 
#     subject_ids: List[str],
#     classifier_name: str
# ) -> Dict:
#     """
#     Leave-One-Subject-Out 交叉驗證
#     """
#     unique_subjects = list(set(subject_ids))
#     n_subjects = len(unique_subjects)
    
#     print(f"    執行 LOSO 交叉驗證，共 {n_subjects} 個獨立受試者")
    
#     all_y_true = []
#     all_y_pred = []
#     all_test_subjects = []
    
#     for i, test_subject in enumerate(unique_subjects):
#         # 建立訓練和測試索引
#         test_indices = [idx for idx, sid in enumerate(subject_ids) if sid == test_subject]
#         train_indices = [idx for idx, sid in enumerate(subject_ids) if sid != test_subject]
        
#         if not test_indices or not train_indices:
#             continue
            
#         X_train = X[train_indices]
#         X_test = X[test_indices]
#         y_train = y[train_indices]
#         y_test = y[test_indices]
        
#         # 標準化（僅對需要的模型）
#         use_scaler = classifier_name in ('SVM', 'Logistic Regression')
        
#         if use_scaler:
#             scaler = StandardScaler()
#             X_train_scaled = scaler.fit_transform(X_train)
#             X_test_scaled = scaler.transform(X_test)
#         else:
#             X_train_scaled, X_test_scaled = X_train, X_test
        
#         # 訓練模型
#         if classifier_name == 'Random Forest':
#             classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
#         elif classifier_name == 'SVM':
#             classifier = SVC(kernel='rbf', random_state=42)
#         elif classifier_name == 'Logistic Regression':
#             classifier = LogisticRegression(max_iter=1000, random_state=42)
#         elif classifier_name == 'XGBoost':
#             classifier = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=5, n_jobs=-1)
        
#         classifier.fit(X_train_scaled, y_train)
        
#         # 預測（對於同一受試者可能有多個樣本，取多數決）
#         y_pred = classifier.predict(X_test_scaled)
        
#         # 如果該受試者有多個樣本，取多數決
#         if len(y_pred) > 1:
#             y_pred_final = 1 if np.mean(y_pred) >= 0.5 else 0
#             y_true_final = 1 if np.mean(y_test) >= 0.5 else 0
#         else:
#             y_pred_final = y_pred[0]
#             y_true_final = y_test[0]
        
#         all_y_true.append(y_true_final)
#         all_y_pred.append(y_pred_final)
#         all_test_subjects.append(test_subject)
        
#         if (i + 1) % 20 == 0:
#             print(f"      進度: {i + 1}/{n_subjects}")
    
#     # 計算整體指標
#     all_y_true = np.array(all_y_true)
#     all_y_pred = np.array(all_y_pred)
    
#     cm = confusion_matrix(all_y_true, all_y_pred)
#     mcc = matthews_corrcoef(all_y_true, all_y_pred)
#     acc = accuracy_score(all_y_true, all_y_pred)
    
#     return {
#         'confusion_matrix': cm,
#         'mcc': mcc,
#         'accuracy': acc,
#         'y_true': all_y_true,
#         'y_pred': all_y_pred,
#         'test_subjects': all_test_subjects
#     }

# def train_with_loso(
#     X: np.ndarray, 
#     y: np.ndarray, 
#     subject_ids: List[str],
#     embedding_model: str, 
#     feature_type: str,
#     output_dir: str = "model_output"
# ) -> Dict:
#     """
#     使用 LOSO 訓練多個分類器
#     """
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     # 加入 Logistic Regression
#     # classifier_names = ['Random Forest', 'XGBoost', 'SVM', 'Logistic Regression']
#     classifier_names = ['Random Forest', 'XGBoost']
    
#     results = {}
#     for classifier_name in classifier_names:
#         print(f"\n  訓練 {classifier_name} ({embedding_model}, {feature_type})...")
        
#         result = leave_one_subject_out_cv(X, y, subject_ids, classifier_name)
        
#         cm = result['confusion_matrix']
#         mcc = result['mcc']
#         acc = result['accuracy']
        
#         results[classifier_name] = {
#             'confusion_matrix': cm.tolist(),
#             'mcc': float(mcc),
#             'accuracy': float(acc),
#             'sensitivity': float(cm[1,1] / (cm[1,0] + cm[1,1])) if (cm[1,0] + cm[1,1]) > 0 else 0,
#             'specificity': float(cm[0,0] / (cm[0,0] + cm[0,1])) if (cm[0,0] + cm[0,1]) > 0 else 0
#         }
        
#         print(f"    準確率: {acc:.4f}")
#         print(f"    MCC: {mcc:.4f}")
#         print(f"    靈敏度: {results[classifier_name]['sensitivity']:.4f}")
#         print(f"    特異度: {results[classifier_name]['specificity']:.4f}")
#         print(f"    混淆矩陣:")
#         print(f"      TN={cm[0,0]}, FP={cm[0,1]}")
#         print(f"      FN={cm[1,0]}, TP={cm[1,1]}")
    
#     return results

# def main():
#     """主程式"""
#     # 基礎路徑設定
#     input_root = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature\datung"
#     output_root = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\results\DeepLearning\6_results_LOSO_aged\datung"
    
#     # 年齡表來源
#     p_csv   = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\p_merged.csv"
#     acs_csv = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\ACS_merged_results.csv"
#     nad_csv = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\NAD_merged_results.csv"
#     excel_file = None
    
#     # 載入年齡表
#     allowed_ids = None
#     demo_lookup = None
#     try:
#         age_tables = load_age_tables(p_source=p_csv, acs_source=acs_csv, nad_source=nad_csv, excel_source=excel_file)
#         allowed_ids, age_summary = age_balance_ids_two_groups(age_tables, nbins=5, seed=42, method="quantile")
#         demo_lookup = build_demo_lookup(age_tables)
#         print("\n年齡配對完成，並建立 Age/Sex 查表。")
#     except Exception as e:
#         print(f"\n[警告] 年齡表載入/配對失敗，將略過年齡/性別特徵：{e}")
    
#     # print("略過年齡/性別")

#     # 設定
#     embedding_models = ['vggface', 'arcface', 'dlib', 'deepid', 'topofr']
#     feature_types = ['difference', 'average', 'relative']  # 三種特徵類型
#     demo_weight = 1.0
    
#     # 總體結果儲存
#     overall_results = {}
    
#     # 處理每組路徑
#     print(f"\n{'='*60}")
    
#     if not os.path.exists(input_root):
#         print(f"⚠️ 警告: 找不到輸入資料夾 '{input_root}'")
#         return
    
#     # 對每個嵌入模型和特徵類型
#     for embedding_model in embedding_models:
#         for feature_type in feature_types:
#             print(f"\n{'='*50}")
#             print(f"處理: {embedding_model} - {feature_type}")
#             print('='*50)
            
#             # 決定是否應用相關係數過濾（僅對方法1和方法2）
#             apply_filter = feature_type in ('difference', 'average')

#             X, y, case_names, subject_ids = load_dataset_for_model(
#                 input_root, 
#                 embedding_model,
#                 feature_type=feature_type,
#                 allowed_ids=allowed_ids,
#                 demo_lookup=demo_lookup,
#                 include_demo=True,
#                 demo_weight=demo_weight,
#                 apply_correlation_filter=apply_filter,
#                 correlation_threshold=0.9
#             )
            
#             if len(X) == 0:
#                 print(f"  警告: {embedding_model}-{feature_type} 沒有載入到任何資料")
#                 continue
            
#             print(f"\n  最終資料集:")
#             print(f"    總樣本數: {len(X)}")
#             print(f"    獨立受試者數: {len(set(subject_ids))}")
#             print(f"    Health: {np.sum(y == 0)}, Patient: {np.sum(y == 1)}")
#             print(f"    特徵維度: {X.shape[1]}")
            
#             # 使用 LOSO 訓練
#             results = train_with_loso(
#                 X, y, subject_ids,
#                 embedding_model, 
#                 feature_type,
#                 output_dir=output_root
#             )
            
#             model_key = f"{embedding_model}_{feature_type}"
#             overall_results[model_key] = results
    
#     # 儲存所有結果
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_filename = Path(output_root) / f"all_loso_results_{timestamp}.json"
#     with open(results_filename, 'w', encoding='utf-8') as f:
#         json.dump(overall_results, f, indent=2, ensure_ascii=False)
    
#     print(f"\n{'='*60}")
#     print("訓練完成！")
#     print(f"結果已儲存至 '{results_filename}'")
    
#     # 總結報告
#     print("\n總結報告 (LOSO):")
#     print("-"*60)
#     for model_key, model_results in overall_results.items():
#         print(f"\n{model_key}:")
#         for classifier_name, metrics in model_results.items():
#             print(f"  {classifier_name}:")
#             print(f"      準確率: {metrics['accuracy']:.4f}")
#             print(f"      MCC: {metrics['mcc']:.4f}")

# if __name__ == "__main__":
#     main()

# ========V3========

# import json
# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple, Optional, Set
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, classification_report
# import xgboost as xgb
# import joblib
# from datetime import datetime
# import re
# import warnings
# warnings.filterwarnings('ignore')

# def parse_case_id(folder_name: str) -> Tuple[str, int]:
#     """
#     解析資料夾名稱，提取個案ID和收案次數
#     例如: P1-2 -> (P1, 2), ACS1-1 -> (ACS1, 1)
#     """
#     patterns = [
#         r'^([A-Za-z]+\d+)-(\d+)$',
#         r'^([A-Za-z]+)-(\d+)$',
#         r'^(\w+)-(\d+)$'
#     ]
#     for pattern in patterns:
#         match = re.match(pattern, folder_name)
#         if match:
#             case_id = match.group(1)
#             visit_number = int(match.group(2))
#             return case_id, visit_number
#     return folder_name, 0

# def filter_latest_cases(case_folders: List[Path]) -> List[Path]:
#     """
#     過濾出每個個案的最後一次收案資料
#     """
#     case_dict = {}
#     for folder in case_folders:
#         folder_name = folder.name
#         case_id, visit_number = parse_case_id(folder_name)
#         if case_id not in case_dict:
#             case_dict[case_id] = []
#         case_dict[case_id].append((visit_number, folder))
#     latest_folders = []
#     for case_id, visits in case_dict.items():
#         visits.sort(key=lambda x: x[0], reverse=True)
#         latest_folders.append(visits[0][1])
#         # if len(visits) > 1:
#         #     print(f"    個案 {case_id}: 找到 {len(visits)} 次收案，選擇第 {visits[0][0]} 次")
#     return latest_folders

# def read_difference_json(filepath: str) -> Dict:
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def extract_features_from_case(case_folder: Path, embedding_model: str, feature_type: str = "difference") -> Optional[np.ndarray]:
#     """
#     從個案資料夾提取指定類型的特徵
#     feature_type: 'difference', 'average', 'relative'
#     """
#     json_files = sorted(case_folder.glob("*_LR_difference.json"))
#     if not json_files:
#         return None
    
#     vectors = []
#     for json_file in json_files:
#         data = read_difference_json(json_file)
        
#         if feature_type == "difference":
#             # 方法0：原始差值特徵
#             feat = data.get("embedding_differences", {}).get(embedding_model)
#         elif feature_type == "average":
#             # 方法1：平均向量特徵
#             feat = data.get("embedding_averages", {}).get(embedding_model)
#         elif feature_type == "relative":
#             # 方法2：相對差異特徵（標量，需要特殊處理）
#             feat = data.get("relative_differences", {}).get(embedding_model)
#             if feat is not None:
#                 # 相對差異是單一數值，不是向量
#                 vectors.append(float(feat))
#                 continue
#         else:
#             raise ValueError(f"Unknown feature_type: {feature_type}")
        
#         if feat is not None:
#             v = np.asarray(feat, dtype=float)
#             vectors.append(v)
    
#     if not vectors:
#         return None
    
#     # 對於相對差異（標量），直接返回平均值
#     if feature_type == "relative":
#         return np.array([np.mean(vectors)])  # 返回一維陣列
    
#     # 對於向量特徵，檢查維度一致性
#     shapes = {v.shape for v in vectors}
#     if len(shapes) > 1:
#         raise ValueError(f"向量長度不一致，無法取平均：{sorted(shapes)}")
#     stacked = np.vstack(vectors)
#     mean_vec = stacked.mean(axis=0)
#     return mean_vec

# def remove_highly_correlated_features(X: np.ndarray, threshold: float = 0.9) -> Tuple[np.ndarray, List[int]]:
#     """
#     移除高度相關的特徵（相關係數 > threshold）
#     返回處理後的特徵矩陣和保留的特徵索引
#     """
#     if X.shape[1] <= 1:
#         return X, list(range(X.shape[1]))
    
#     # 計算相關係數矩陣
#     corr_matrix = np.corrcoef(X.T)
    
#     # 找出要保留的特徵
#     keep_features = []
#     removed_features = set()
    
#     for i in range(corr_matrix.shape[0]):
#         if i in removed_features:
#             continue
#         keep_features.append(i)
#         # 找出與當前特徵高度相關的其他特徵
#         for j in range(i+1, corr_matrix.shape[1]):
#             if abs(corr_matrix[i, j]) > threshold:
#                 removed_features.add(j)
    
#     print(f"    原始特徵數: {X.shape[1]}, 移除 {len(removed_features)} 個高度相關特徵, 保留 {len(keep_features)} 個")
    
#     # 只保留選中的特徵
#     X_filtered = X[:, keep_features]
#     return X_filtered, keep_features

# # ==============================
# # === 年齡表與配對功能（保持不變） ===
# # ==============================

# def _select_latest_rows_by_id(df: pd.DataFrame) -> pd.DataFrame:
#     """將同一 base_id 只保留最後一次收案"""
#     df = df.copy()
#     df["base_id"] = df["ID"].apply(lambda s: parse_case_id(str(s))[0])
#     df["visit_number"] = df["ID"].apply(lambda s: parse_case_id(str(s))[1])

#     def pick_last(group: pd.DataFrame) -> pd.Series:
#         g = group.sort_values(
#             by=["visit_number", "Photo_Session"] if "Photo_Session" in group.columns else ["visit_number"],
#             ascending=False
#         )
#         return g.iloc[0]

#     latest = df.groupby("base_id", as_index=False, group_keys=False).apply(pick_last)
#     latest = latest.drop(columns=["base_id", "visit_number"])
#     return latest

# def _read_age_table_generic(path: str) -> pd.DataFrame:
#     """讀取年齡表（保持原樣）"""
#     p = Path(path)

#     if p.suffix.lower() in [".xlsx", ".xls"]:
#         df = pd.read_excel(p)
#     else:
#         tried_encs = []
#         df = None
#         for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
#             try:
#                 df = pd.read_csv(p, sep=None, engine="python", encoding=enc)
#                 break
#             except Exception as e:
#                 tried_encs.append(f"{enc}: {e}")
#         if df is None:
#             raise RuntimeError(f"讀取 CSV 失敗（嘗試編碼：{'; '.join(tried_encs)}）")

#     def _normalize_col(s: str) -> str:
#         s = str(s)
#         s = s.replace("\ufeff", "").replace("\u200b", "")
#         s = s.replace("（", "(").replace("）", ")")
#         s = s.strip()
#         s = re.sub(r"\s+", "", s)
#         return s.lower()

#     raw_cols = list(df.columns)
#     norm_cols = {_normalize_col(c): c for c in df.columns}

#     id_keys  = {"ID"}
#     age_keys = {"Age"}
#     sex_keys = {"Sex"}
#     cdr_keys = {"Global_CDR"}  # 新增 CDR 欄位識別

#     rename_map = {}
#     found_id_src = found_age_src = found_sex_src = found_cdr_src = None

#     for norm, src in norm_cols.items():
#         if norm in id_keys and found_id_src is None:
#             rename_map[src] = "ID"; found_id_src = src
#         if norm in age_keys and found_age_src is None:
#             rename_map[src] = "Age"; found_age_src = src
#         if norm in sex_keys and found_sex_src is None:
#             rename_map[src] = "Sex"; found_sex_src = src
#         if norm in cdr_keys and found_cdr_src is None:
#             rename_map[src] = "Global_CDR"; found_cdr_src = src

#     df = df.rename(columns=rename_map)

#     if "ID" not in df.columns or "Age" not in df.columns:
#         debug_msg = (
#             f"{path} 缺少必要欄位：ID 或 Age\n"
#             f"實際讀到欄位：{raw_cols}\n"
#             f"正規化後欄位：{[_normalize_col(c) for c in raw_cols]}"
#         )
#         raise ValueError(debug_msg)

#     df = df.dropna(subset=["ID", "Age"])
#     df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
#     df = df.dropna(subset=["Age"])
    
#     # 處理 CDR 欄位
#     if "Global_CDR" in df.columns:
#         df["Global_CDR"] = pd.to_numeric(df["Global_CDR"], errors="coerce")
#         print(f"  讀取到 Global_CDR 欄位，有效值範圍: {df['Global_CDR'].min():.1f} - {df['Global_CDR'].max():.1f}")
    
#     # 處理性別欄位
#     if "Sex" in df.columns:
#         def _parse_sex(x):
#             if pd.isna(x): return np.nan
#             s = str(x).strip().lower()
#             if s in {"m", "male", "man", "boy", "1", "男", "男性"}: return 1
#             if s in {"f", "female", "woman", "girl", "0", "女", "女性"}: return 0
#             return np.nan
#         df["Sex"] = df["Sex"].apply(_parse_sex)

#     return df

# def load_age_tables(
#     p_source: Optional[str] = None,
#     acs_source: Optional[str] = None,
#     nad_source: Optional[str] = None,
#     excel_source: Optional[str] = None
# ) -> Dict[str, pd.DataFrame]:
#     """載入年齡表"""
#     if excel_source:
#         xl = pd.ExcelFile(excel_source)
#         sheets = {name.lower(): name for name in xl.sheet_names}
#         def pick_sheet(keyword: str) -> str:
#             for k, v in sheets.items():
#                 if keyword.lower() in k:
#                     return v
#             raise ValueError(f"Excel 中找不到包含 '{keyword}' 的工作表")
#         p_df = pd.read_excel(excel_source, sheet_name=pick_sheet("P"))
#         acs_df = pd.read_excel(excel_source, sheet_name=pick_sheet("ACS"))
#         nad_df = pd.read_excel(excel_source, sheet_name=pick_sheet("NAD"))
#     else:
#         if not (p_source and acs_source and nad_source):
#             raise ValueError("請提供 excel_source，或同時提供 p_source、acs_source、nad_source 三個路徑")
#         p_df = _read_age_table_generic(p_source)
#         acs_df = _read_age_table_generic(acs_source)
#         nad_df = _read_age_table_generic(nad_source)

#     p_df = _select_latest_rows_by_id(p_df)
#     acs_df = _select_latest_rows_by_id(acs_df)
#     nad_df = _select_latest_rows_by_id(nad_df)

#     return {"P": p_df, "ACS": acs_df, "NAD": nad_df}

# def filter_by_cdr(tables: Dict[str, pd.DataFrame], cdr_threshold: float = None) -> Dict[str, pd.DataFrame]:
#     """
#     根據 Global_CDR 值篩選 P 組資料
#     cdr_threshold: CDR 閾值，只保留 CDR > threshold 的資料
#     """
#     filtered_tables = tables.copy()
    
#     if cdr_threshold is not None and "Global_CDR" in tables["P"].columns:
#         original_count = len(tables["P"])
#         # 篩選 CDR > threshold
#         filtered_p = tables["P"][tables["P"]["Global_CDR"] > cdr_threshold].copy()
#         filtered_tables["P"] = filtered_p
#         filtered_count = len(filtered_p)
        
#         print(f"\n[CDR 篩選結果]")
#         print(f"  CDR 閾值: > {cdr_threshold}")
#         print(f"  P組原始數量: {original_count}")
#         print(f"  篩選後數量: {filtered_count}")
#         print(f"  保留比例: {filtered_count/original_count*100:.1f}%")
#     elif cdr_threshold is not None:
#         print(f"\n[警告] P 表中沒有 Global_CDR 欄位，跳過 CDR 篩選")
    
#     return filtered_tables

# def age_balance_ids_two_groups(
#     tables: Dict[str, pd.DataFrame],
#     nbins: int = 5,
#     seed: int = 42,
#     method: str = "quantile",
#     enable_age_matching: bool = True,  # 新增：是否啟用年齡配對
#     cdr_threshold: float = None        # CDR 閾值
# ) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
#     """
#     兩群年齡配對（Health vs P）
#     enable_age_matching: 是否進行年齡配對，若為False則返回所有符合CDR條件的ID
#     cdr_threshold: CDR閾值，只保留P組中CDR > threshold的資料
#     """
#     rng = np.random.RandomState(seed)

#     # Step 1: 先進行 CDR 篩選（如果指定）
#     if cdr_threshold is not None:
#         tables = filter_by_cdr(tables, cdr_threshold)
        
#         # 如果篩選後 P 組沒有資料，返回空結果
#         if len(tables["P"]) == 0:
#             print(f"  警告：CDR > {cdr_threshold} 篩選後，P 組沒有剩餘資料")
#             return {"ACS": set(), "NAD": set(), "P": set()}, pd.DataFrame()

#     # 準備資料
#     acs_df = tables["ACS"][["ID", "Age"]].copy()
#     acs_df["origin"] = "ACS"
#     nad_df = tables["NAD"][["ID", "Age"]].copy()
#     nad_df["origin"] = "NAD"
#     p_df   = tables["P"][["ID", "Age"]].copy()

#     health_df = pd.concat([acs_df, nad_df], ignore_index=True)
#     health_df["group2"] = "Health"
#     p_df["group2"] = "P"

#     all_df = pd.concat([health_df[["ID","Age","group2","origin"]],
#                         p_df[["ID","Age","group2"]]], ignore_index=True)

#     # Step 2: 根據是否啟用年齡配對來處理
#     if not enable_age_matching:
#         # 不進行年齡配對，直接返回所有ID
#         print("\n[跳過年齡配對]")
        
#         # 直接使用所有的ID
#         acs_ids = set(tables["ACS"]["ID"].tolist())
#         nad_ids = set(tables["NAD"]["ID"].tolist())
#         p_ids = set(tables["P"]["ID"].tolist())
        
#         allowed_ids = {
#             "ACS": acs_ids,
#             "NAD": nad_ids,
#             "P": p_ids
#         }
        
#         # 計算統計資料
#         stats = []
#         for g_name, ids in [("Health", acs_ids.union(nad_ids)), ("P", p_ids)]:
#             g_sub = all_df[all_df["ID"].isin(ids)]
#             stats.append({
#                 "group": g_name,
#                 "n_selected": int(g_sub.shape[0]),
#                 "age_mean": float(g_sub["Age"].mean()) if not g_sub.empty else np.nan,
#                 "age_std": float(g_sub["Age"].std(ddof=1)) if g_sub.shape[0] > 1 else np.nan
#             })
#         summary = pd.DataFrame(stats)
        
#         cdr_str = f" (CDR > {cdr_threshold})" if cdr_threshold is not None else ""
#         print(f"\n[統計資料（無年齡配對）{cdr_str}]")
#         for _, row in summary.iterrows():
#             if pd.notnull(row["age_mean"]):
#                 print(f"  {row['group']}: n={row['n_selected']}, mean={row['age_mean']:.2f}, std={row['age_std']:.2f}")
#             else:
#                 print(f"  {row['group']}: n={row['n_selected']}")
#         print(f"  Health 組內：ACS={len(acs_ids)}, NAD={len(nad_ids)}")
        
#         return allowed_ids, summary
    
#     # Step 3: 進行年齡配對（原有邏輯）
#     print("\n[執行年齡配對]")
    
#     if method == "quantile":
#         try:
#             all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins, duplicates="drop")
#         except ValueError:
#             nbins_eff = max(2, min(nbins, all_df["Age"].nunique()))
#             all_df["age_bin"] = pd.qcut(all_df["Age"], q=nbins_eff, duplicates="drop")
#     else:
#         raise NotImplementedError("目前僅支援 method='quantile'")

#     selected_health_ids: Set[str] = set()
#     selected_p_ids: Set[str] = set()

#     for b in all_df["age_bin"].cat.categories:
#         bin_df = all_df[all_df["age_bin"] == b]
#         n_health = bin_df[bin_df["group2"] == "Health"].shape[0]
#         n_p      = bin_df[bin_df["group2"] == "P"].shape[0]
#         target = min(n_health, n_p)
#         if target == 0:
#             continue

#         health_pool = bin_df[bin_df["group2"] == "Health"]
#         p_pool      = bin_df[bin_df["group2"] == "P"]

#         pick_h = health_pool.sample(n=target, random_state=rng) if len(health_pool) >= target else health_pool
#         pick_p = p_pool.sample(n=target, random_state=rng) if len(p_pool) >= target else p_pool

#         selected_health_ids.update(pick_h["ID"].tolist())
#         selected_p_ids.update(pick_p["ID"].tolist())

#     acs_ids_all = set(tables["ACS"]["ID"].tolist())
#     nad_ids_all = set(tables["NAD"]["ID"].tolist())

#     selected_acs = set([i for i in selected_health_ids if i in acs_ids_all])
#     selected_nad = set([i for i in selected_health_ids if i in nad_ids_all])

#     allowed_ids = {
#         "ACS": selected_acs,
#         "NAD": selected_nad,
#         "P":   selected_p_ids
#     }

#     stats = []
#     for g_name, ids in [("Health", selected_health_ids), ("P", selected_p_ids)]:
#         g_sub = all_df[all_df["ID"].isin(ids)]
#         stats.append({
#             "group": g_name,
#             "n_selected": int(g_sub.shape[0]),
#             "age_mean": float(g_sub["Age"].mean()) if not g_sub.empty else np.nan,
#             "age_std": float(g_sub["Age"].std(ddof=1)) if g_sub.shape[0] > 1 else np.nan
#         })
#     summary = pd.DataFrame(stats)

#     cdr_str = f" (CDR > {cdr_threshold})" if cdr_threshold is not None else ""
#     print(f"\n[年齡配對後（Health vs P）的統計{cdr_str}]")
#     for _, row in summary.iterrows():
#         if pd.notnull(row["age_mean"]):
#             print(f"  {row['group']}: n={row['n_selected']}, mean={row['age_mean']:.2f}, std={row['age_std']:.2f}")
#         else:
#             print(f"  {row['group']}: n={row['n_selected']}")

#     print(f"  Health 組內：ACS={len(selected_acs)}, NAD={len(selected_nad)}")

#     return allowed_ids, summary

# def build_demo_lookup(tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
#     """建立人口學資料查詢表"""
#     frames = []
#     for g in ["P", "ACS", "NAD"]:
#         df = tables[g][["ID", "Age"]].copy()
#         if "Sex" in tables[g].columns:
#             df["Sex"] = tables[g]["Sex"]
#         frames.append(df)
#     merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ID"], keep="first")

#     lookup = {}
#     for _, row in merged.iterrows():
#         lookup[str(row["ID"])] = {
#             "Age": float(row["Age"]),
#             "Sex": (float(row["Sex"]) if "Sex" in merged.columns and pd.notna(row["Sex"]) else np.nan)
#         }
#     if "Sex" in merged.columns:
#         sex_mode = merged["Sex"].dropna().mode()
#         lookup["_SEX_MODE_"] = float(sex_mode.iloc[0]) if len(sex_mode)>0 else np.nan
#     else:
#         lookup["_SEX_MODE_"] = np.nan
#     return lookup

# def load_dataset_for_model(
#     data_root: str,
#     embedding_model: str,
#     feature_type: str = "difference",  # 新增：特徵類型選擇
#     allowed_ids: Optional[Dict[str, Set[str]]] = None,
#     demo_lookup: Optional[Dict[str, Dict[str, float]]] = None, 
#     include_demo: bool = True,
#     demo_weight: float = 1.0,
#     apply_correlation_filter: bool = False,  # 新增：是否應用相關係數過濾
#     correlation_threshold: float = 0.9       # 新增：相關係數閾值
# ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:  # 新增返回subject_ids
#     """
#     載入資料集並提取特徵
#     返回：X, y, case_names, subject_ids（用於LOSO）
#     """
#     data_path = Path(data_root)
#     X_list, y_list, case_names = [], [], []
#     subject_ids = []  # 新增：記錄每個樣本的subject ID
#     demo_list: List[Tuple[float, float]] = []

#     def _maybe_push_demo(case_id: str):
#         if not include_demo or demo_lookup is None: 
#             return
#         meta = demo_lookup.get(case_id)
#         if meta is None:
#             demo_list.append((np.nan, np.nan)); return
#         age = meta["Age"]
#         sex = meta["Sex"]
#         if pd.isna(sex):
#             sex = demo_lookup.get("_SEX_MODE_", np.nan)
#         demo_list.append((age, sex))

#     # --- health 類別
#     health_path = data_path / "health"
#     if health_path.exists():
#         print(f"  處理 health 類別 ({embedding_model}, {feature_type})...")
#         for category_folder in health_path.iterdir():
#             if category_folder.is_dir():
#                 group_name = category_folder.name
#                 all_case_folders = [f for f in category_folder.iterdir() if f.is_dir()]
#                 latest_case_folders = filter_latest_cases(all_case_folders)
#                 if allowed_ids and group_name in allowed_ids:
#                     before = len(latest_case_folders)
#                     latest_case_folders = [f for f in latest_case_folders if f.name in allowed_ids[group_name]]
#                     print(f"    {category_folder.name}: {len(latest_case_folders)} 個個案（原 {before}，年齡篩選後）")
#                 else:
#                     print(f"    {category_folder.name}: {len(latest_case_folders)} 個個案（無年齡篩選）")

#                 for case_folder in latest_case_folders:
#                     features = extract_features_from_case(case_folder, embedding_model, feature_type)
#                     if features is not None:
#                         X_list.append(features)
#                         y_list.append(0)
#                         case_names.append(f"health/{category_folder.name}/{case_folder.name}")
#                         subject_ids.append(case_folder.name)  # 記錄subject ID
#                         _maybe_push_demo(case_folder.name)

#     # --- patient 類別
#     patient_path = data_path / "patient"
#     if patient_path.exists():
#         print(f"  處理 patient 類別 ({embedding_model}, {feature_type})...")
#         all_case_folders = [f for f in patient_path.iterdir() if f.is_dir()]
#         latest_case_folders = filter_latest_cases(all_case_folders)
#         if allowed_ids and "P" in allowed_ids:
#             before = len(latest_case_folders)
#             latest_case_folders = [f for f in latest_case_folders if f.name in allowed_ids["P"]]
#             print(f"    找到 {len(latest_case_folders)} 個個案（原 {before}，年齡篩選後）")
#         else:
#             print(f"    找到 {len(latest_case_folders)} 個個案（無年齡篩選）")

#         for case_folder in latest_case_folders:
#             features = extract_features_from_case(case_folder, embedding_model, feature_type)
#             if features is not None:
#                 X_list.append(features)
#                 y_list.append(1)
#                 case_names.append(f"patient/{case_folder.name}")
#                 subject_ids.append(case_folder.name)  # 記錄subject ID
#                 _maybe_push_demo(case_folder.name)

#     # --- 尺寸對齊
#     if X_list:
#         if feature_type != "relative":  # 相對差異是標量，不需要對齊
#             max_len = max(len(x) for x in X_list)
#             X_list_padded = []
#             for x in X_list:
#                 if len(x) < max_len:
#                     X_list_padded.append(np.pad(x, (0, max_len - len(x)), 'constant', constant_values=0))
#                 elif len(x) > max_len:
#                     X_list_padded.append(x[:max_len])
#                 else:
#                     X_list_padded.append(x)
#             X_embed = np.asarray(X_list_padded, dtype=np.float64)
#         else:
#             X_embed = np.asarray(X_list, dtype=np.float64)
#             if len(X_embed.shape) == 1:
#                 X_embed = X_embed.reshape(-1, 1)
#     else:
#         return np.array([]), np.array([]), [], []

#     y = np.asarray(y_list, dtype=int)

#     # --- 應用相關係數過濾（僅對高維特徵）
#     if apply_correlation_filter and feature_type != "relative" and X_embed.shape[1] > 1:
#         print(f"  應用相關係數過濾 (閾值={correlation_threshold})...")
#         X_embed, kept_features = remove_highly_correlated_features(X_embed, correlation_threshold)

#     # --- 標準化與串接 demo
#     if include_demo and demo_lookup is not None:
#         demo_arr = np.array(demo_list, dtype=np.float64)
#         # 補值處理
#         if np.isnan(demo_arr[:,0]).any():
#             age_mean = np.nanmean(demo_arr[:,0])
#             demo_arr[np.isnan(demo_arr[:,0]), 0] = age_mean
#         if np.isnan(demo_arr[:,1]).any():
#             sex_mode = demo_lookup.get("_SEX_MODE_", np.nan)
#             fill_val = sex_mode if not np.isnan(sex_mode) else 0.5
#             demo_arr[np.isnan(demo_arr[:,1]), 1] = fill_val

#         # 標準化
#         embed_mean = X_embed.mean(axis=0, keepdims=True)
#         embed_std  = X_embed.std(axis=0, keepdims=True)
#         embed_std[embed_std == 0] = 1.0
#         X_embed_z = (X_embed - embed_mean) / embed_std

#         demo_mean = demo_arr.mean(axis=0, keepdims=True)
#         demo_std  = demo_arr.std(axis=0, keepdims=True)
#         demo_std[demo_std == 0] = 1.0
#         demo_z = (demo_arr - demo_mean) / demo_std

#         X = np.hstack([X_embed_z, demo_weight * demo_z])
#     else:
#         X = X_embed

#     return X, y, case_names, subject_ids

# def leave_one_subject_out_cv(
#     X: np.ndarray, 
#     y: np.ndarray, 
#     subject_ids: List[str],
#     classifier_name: str
# ) -> Dict:
#     """
#     Leave-One-Subject-Out 交叉驗證
#     """
#     unique_subjects = list(set(subject_ids))
#     n_subjects = len(unique_subjects)
    
#     print(f"    執行 LOSO 交叉驗證，共 {n_subjects} 個獨立受試者")
    
#     all_y_true = []
#     all_y_pred = []
#     all_test_subjects = []
    
#     for i, test_subject in enumerate(unique_subjects):
#         # 建立訓練和測試索引
#         test_indices = [idx for idx, sid in enumerate(subject_ids) if sid == test_subject]
#         train_indices = [idx for idx, sid in enumerate(subject_ids) if sid != test_subject]
        
#         if not test_indices or not train_indices:
#             continue
            
#         X_train = X[train_indices]
#         X_test = X[test_indices]
#         y_train = y[train_indices]
#         y_test = y[test_indices]
        
#         # 標準化（僅對需要的模型）
#         use_scaler = classifier_name in ('SVM', 'Logistic Regression')
        
#         if use_scaler:
#             scaler = StandardScaler()
#             X_train_scaled = scaler.fit_transform(X_train)
#             X_test_scaled = scaler.transform(X_test)
#         else:
#             X_train_scaled, X_test_scaled = X_train, X_test
        
#         # 訓練模型
#         if classifier_name == 'Random Forest':
#             classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
#         elif classifier_name == 'SVM':
#             classifier = SVC(kernel='rbf', random_state=42)
#         elif classifier_name == 'Logistic Regression':
#             classifier = LogisticRegression(max_iter=1000, random_state=42)
#         elif classifier_name == 'XGBoost':
#             classifier = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=5, n_jobs=-1)
        
#         classifier.fit(X_train_scaled, y_train)
        
#         # 預測（對於同一受試者可能有多個樣本，取多數決）
#         y_pred = classifier.predict(X_test_scaled)
        
#         # 如果該受試者有多個樣本，取多數決
#         if len(y_pred) > 1:
#             y_pred_final = 1 if np.mean(y_pred) >= 0.5 else 0
#             y_true_final = 1 if np.mean(y_test) >= 0.5 else 0
#         else:
#             y_pred_final = y_pred[0]
#             y_true_final = y_test[0]
        
#         all_y_true.append(y_true_final)
#         all_y_pred.append(y_pred_final)
#         all_test_subjects.append(test_subject)
        
#         if (i + 1) % 20 == 0:
#             print(f"      進度: {i + 1}/{n_subjects}")
    
#     # 計算整體指標
#     all_y_true = np.array(all_y_true)
#     all_y_pred = np.array(all_y_pred)
    
#     cm = confusion_matrix(all_y_true, all_y_pred)
#     mcc = matthews_corrcoef(all_y_true, all_y_pred)
#     acc = accuracy_score(all_y_true, all_y_pred)
    
#     return {
#         'confusion_matrix': cm,
#         'mcc': mcc,
#         'accuracy': acc,
#         'y_true': all_y_true,
#         'y_pred': all_y_pred,
#         'test_subjects': all_test_subjects
#     }

# def train_with_loso(
#     X: np.ndarray, 
#     y: np.ndarray, 
#     subject_ids: List[str],
#     embedding_model: str, 
#     feature_type: str,
#     output_dir: str = "model_output"
# ) -> Dict:
#     """
#     使用 LOSO 訓練多個分類器
#     """
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     # 加入 Logistic Regression
#     classifier_names = ['Random Forest', 'XGBoost', 'SVM', 'Logistic Regression']
    
#     results = {}
#     for classifier_name in classifier_names:
#         print(f"\n  訓練 {classifier_name} ({embedding_model}, {feature_type})...")
        
#         result = leave_one_subject_out_cv(X, y, subject_ids, classifier_name)
        
#         cm = result['confusion_matrix']
#         mcc = result['mcc']
#         acc = result['accuracy']
        
#         results[classifier_name] = {
#             'confusion_matrix': cm.tolist(),
#             'mcc': float(mcc),
#             'accuracy': float(acc),
#             'sensitivity': float(cm[1,1] / (cm[1,0] + cm[1,1])) if (cm[1,0] + cm[1,1]) > 0 else 0,
#             'specificity': float(cm[0,0] / (cm[0,0] + cm[0,1])) if (cm[0,0] + cm[0,1]) > 0 else 0
#         }
        
#         print(f"    準確率: {acc:.4f}")
#         print(f"    MCC: {mcc:.4f}")
#         print(f"    靈敏度: {results[classifier_name]['sensitivity']:.4f}")
#         print(f"    特異度: {results[classifier_name]['specificity']:.4f}")
#         print(f"    混淆矩陣:")
#         print(f"      TN={cm[0,0]}, FP={cm[0,1]}")
#         print(f"      FN={cm[1,0]}, TP={cm[1,1]}")
    
#     return results

# def main():
#     """主程式"""
#     # 基礎路徑設定
#     input_root = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature\datung"
#     output_root = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\results\DeepLearning\20250923_LOSO_no_aged_cdr_filtered"

#     # 年齡表來源
#     p_csv   = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\p_merged.csv"
#     acs_csv = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\ACS_merged_results.csv"
#     nad_csv = r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\NAD_merged_results.csv"
#     excel_file = None
    
#     # ============================
#     # 控制參數設定
#     # ============================
#     ENABLE_AGE_MATCHING = False  # 設為 True 啟用年齡配對，False 停用
#     ENABLE_CDR_FILTER = True    # 設為 True 啟用 CDR 篩選，False 停用
#     CDR_THRESHOLDS = [0.5, 1, 2] if ENABLE_CDR_FILTER else [None]  # CDR 閾值列表
    
#     print("\n" + "="*60)
#     print("執行設定:")
#     print(f"  年齡配對: {'啟用' if ENABLE_AGE_MATCHING else '停用'}")
#     print(f"  CDR篩選: {'啟用' if ENABLE_CDR_FILTER else '停用'}")
#     if ENABLE_CDR_FILTER:
#         print(f"  CDR閾值: {CDR_THRESHOLDS}")
#     print("="*60)

#     # 總體結果儲存
#     overall_results = {}

#     for cdr_threshold in CDR_THRESHOLDS:
#         cdr_results = {}

#         if ENABLE_CDR_FILTER:
#             print(f"\n{'#'*60}")
#             print(f"使用 CDR 閾值 > {cdr_threshold} 進行篩選")
#             print(f"{'#'*60}")
#         else:
#             print(f"\n{'#'*60}")
#             print(f"不使用 CDR 篩選")
#             print(f"{'#'*60}")
        
#         # 載入並配對年齡表
#         allowed_ids = None
#         demo_lookup = None
        
#         try:
#             age_tables = load_age_tables(
#                 p_source=p_csv, 
#                 acs_source=acs_csv, 
#                 nad_source=nad_csv, 
#                 excel_source=excel_file
#             )
            
#             # 使用新的參數來控制年齡配對和CDR篩選
#             allowed_ids, age_summary = age_balance_ids_two_groups(
#                 age_tables, 
#                 nbins=5, 
#                 seed=42, 
#                 method="quantile",
#                 enable_age_matching=ENABLE_AGE_MATCHING,  # 使用控制參數
#                 cdr_threshold=cdr_threshold if ENABLE_CDR_FILTER else None
#             )
            
#             demo_lookup = build_demo_lookup(age_tables)
            
#             if ENABLE_AGE_MATCHING:
#                 print("\n年齡配對完成，並建立 Age/Sex 查表。")
#             else:
#                 print("\n跳過年齡配對，僅建立 Age/Sex 查表。")
                
#         except Exception as e:
#             print(f"\n[警告] 處理失敗：{e}")
#             continue
        
#         # 設定
#         embedding_models = ['vggface', 'arcface', 'dlib', 'deepid', 'topofr']
#         feature_types = ['difference', 'average', 'relative']  # 三種特徵類型
#         demo_weight = 1.0
        
#         # 對每個嵌入模型和特徵類型
#         for embedding_model in embedding_models:
#             for feature_type in feature_types:
#                 print(f"\n{'='*50}")
#                 print(f"處理: {embedding_model} - {feature_type}")
#                 print('='*50)
                
#                 # 決定是否應用相關係數過濾（僅對方法1和方法2）
#                 apply_filter = (feature_type in ['average', 'relative'] and feature_type != 'relative')
                
#                 X, y, case_names, subject_ids = load_dataset_for_model(
#                     input_root, 
#                     embedding_model,
#                     feature_type=feature_type,
#                     allowed_ids=allowed_ids,
#                     demo_lookup=demo_lookup,
#                     include_demo=True,
#                     demo_weight=demo_weight,
#                     apply_correlation_filter=apply_filter,
#                     correlation_threshold=0.9
#                 )
                
#                 if len(X) == 0:
#                     print(f"  警告: {embedding_model}-{feature_type} 沒有載入到任何資料")
#                     continue
                
#                 print(f"\n  最終資料集:")
#                 print(f"    總樣本數: {len(X)}")
#                 print(f"    獨立受試者數: {len(set(subject_ids))}")
#                 print(f"    Health: {np.sum(y == 0)}, Patient: {np.sum(y == 1)}")
#                 print(f"    特徵維度: {X.shape[1]}")
                
#                 # 使用 LOSO 訓練
#                 results = train_with_loso(
#                     X, y, subject_ids,
#                     embedding_model, 
#                     feature_type,
#                     output_dir=output_root
#                 )
                
#                 model_key = f"{embedding_model}_{feature_type}"
#                 cdr_results[model_key] = results

#         # 根據設定決定結果的鍵名
#         if ENABLE_CDR_FILTER:
#             result_key = f"CDR_gt_{cdr_threshold}"
#         else:
#             result_key = "No_filtering"
            
#         overall_results[result_key] = cdr_results

#     # 儲存所有結果
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # 根據設定調整輸出檔名
#     suffix_parts = []
#     if not ENABLE_AGE_MATCHING:
#         suffix_parts.append("no_age")
#     if ENABLE_CDR_FILTER:
#         suffix_parts.append("cdr_filter")
#     suffix = "_".join(suffix_parts) if suffix_parts else "default"
    
#     results_filename = Path(output_root) / f"all_loso_results_{suffix}_{timestamp}.json"
    
#     with open(results_filename, 'w', encoding='utf-8') as f:
#         json.dump(overall_results, f, indent=2, ensure_ascii=False)
    
#     print(f"\n{'='*60}")
#     print("訓練完成！")
#     print(f"結果已儲存至 '{results_filename}'")
    
#     # 總結報告
#     print("\n總結報告 (LOSO):")
#     print("-"*60)
#     for cdr_key, models in overall_results.items():
#         print(f"\n篩選條件: {cdr_key}")
#         for model_key, clf_dict in models.items():
#             print(f"  {model_key}:")
#             for clf_name, metrics in clf_dict.items():
#                 acc = metrics.get('accuracy')
#                 mcc = metrics.get('mcc')
#                 if acc is None or mcc is None:
#                     print(f"    {clf_name}: (缺少指標)")
#                     continue
#                 print(f"    {clf_name}:")
#                 print(f"      準確率: {acc:.4f}")
#                 print(f"      MCC: {mcc:.4f}")

# if __name__ == "__main__":
#     main()

# ========V4========

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