# scripts/test_data_loading.py
"""測試資料載入"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loader import FeatureDataLoader

def main():
    # 載入資料
    loader = FeatureDataLoader(
        r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature_V2\datung"
    )
    
    # 掃描個案
    subjects = loader.scan_subjects(use_all_visits=False)
    print(f"找到 {len(subjects)} 個個案")

    # 載入 VGGFace 差異特徵
    feature_data = loader.load_features(
        subjects, 
        embedding_model='vggface',
        feature_type='difference'
    )
    
    # 顯示統計
    info = loader.get_dataset_info(feature_data)
    print(f"""
    資料集統計:
    - 樣本數: {info.n_samples}
    - 受試者數: {info.n_subjects}  
    - 健康組: {info.n_health}
    - 病患組: {info.n_patient}
    - 各群組: {info.groups}
    - 特徵維度: {info.feature_dim}
    """)

if __name__ == '__main__':
    main()