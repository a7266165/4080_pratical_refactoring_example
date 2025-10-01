# config/path_config.py
"""路徑配置檔"""
from pathlib import Path

# 基礎路徑
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 資料路徑配置
DATA_PATHS = {
    # 人口學資料
    "demographics": {
        "p_csv": DATA_DIR / "demographics" / "p_merged.csv",
        "acs_csv": DATA_DIR / "demographics" / "ACS_merged_results.csv", 
        "nad_csv": DATA_DIR / "demographics" / "NAD_merged_results.csv"
    },
    
    # 圖片資料
    "images": {
        "raw": Path(r"D:\project\Alz\face\data\datung\raw"),  # 外部路徑
        "preprocessed": DATA_DIR / "images" / "preprocessed"   # 專案內
    },
    # 特徵資料
    "features": {
        "datung": DATA_DIR / "features" / "datung",
        "datung_legacy": Path(r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature_V2\datung")
    }
}

