# config/path_config.py
"""路徑配置檔"""
from pathlib import Path

# 基礎路徑
BASE_DIR = Path(r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry")

# 資料路徑
DATA_PATHS = {
    # 原始圖片路徑
    "raw_images": BASE_DIR / "data" / "_pics" / "0_raw",
    
    # 處理過的圖片路徑
    "processed_images": {
        "selected": BASE_DIR / "data" / "_pics" / "1_selected",
        "aligned": BASE_DIR / "data" / "_pics" / "2_aligned",
        "mirrored": BASE_DIR / "data" / "_pics" / "3_mirrored",
        "histogram_matched": BASE_DIR / "data" / "_pics" / "3_histogram_matched"
    },
    
    # 特徵向量路徑
    "features": BASE_DIR / "data" / "features" / "datung" / "DeepLearning" / "5_vector_to_feature_V2" / "datung",
    
    # 人口學資料路徑
    "demographics": {
        "p_csv": r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\p_merged.csv",
        "acs_csv": r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\ACS_merged_results.csv",
        "nad_csv": r"D:\project\Alz\face\data\datung\拍攝日期對應問卷\NAD_merged_results.csv"
    }
}

# 輸出路徑
OUTPUT_PATHS = {
    "results": BASE_DIR / "results" / "new_pipeline",
    "logs": BASE_DIR / "logs"
}

# 模型相關路徑
MODEL_PATHS = {
    "topofr": {
        "path": r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\code\DeepLearning\TopoFR",
        "model": "Glint360K_R100_TopoFR_9760.pt"
    }
}

def get_data_path(key: str) -> Path:
    """獲取資料路徑"""
    if key in DATA_PATHS:
        path = DATA_PATHS[key]
        if isinstance(path, dict):
            raise ValueError(f"'{key}' 包含多個子路徑，請指定具體路徑")
        return Path(path)
    raise KeyError(f"找不到路徑: {key}")

def get_demo_path(key: str) -> str:
    """獲取人口學資料路徑"""
    if key in DATA_PATHS["demographics"]:
        return DATA_PATHS["demographics"][key]
    raise KeyError(f"找不到人口學資料路徑: {key}")

def ensure_output_dirs():
    """確保輸出目錄存在"""
    for path in OUTPUT_PATHS.values():
        Path(path).mkdir(parents=True, exist_ok=True)