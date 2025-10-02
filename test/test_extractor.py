# test/test_extractor.py
"""測試特徵提取模組"""
import sys
from pathlib import Path
import json

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config.path_config as path_config
import src.extractor as extractor_module

import logging
from datetime import datetime
import numpy as np

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 從模組取得需要的類別和變數
DATA_PATHS = path_config.DATA_PATHS
FeatureExtractor = extractor_module.FeatureExtractor
batch_extract_features = extractor_module.batch_extract_features


def test_extractor():
    """測試特徵提取器"""
    
    # 路徑設定
    input_dir = DATA_PATHS["images"]["preprocessed"]  # 從前處理的輸出讀取
    output_dir = DATA_PATHS["features"]["datung"]
    
    logger.info("="*60)
    logger.info("開始測試特徵提取器")
    logger.info("="*60)
    logger.info(f"輸入路徑: {input_dir}")
    logger.info(f"輸出路徑: {output_dir}")
    
    # 檢查輸入路徑
    if not input_dir.exists():
        logger.error(f"輸入路徑不存在: {input_dir}")
        logger.error("請先執行 test_preprocessor.py 產生處理後的圖片")
        return
    
    # 檢查是否有左右臉配對
    left_mirrors = list(input_dir.rglob("*_Lmirror_claheL.png"))
    right_mirrors = list(input_dir.rglob("*_Rmirror_claheL.png"))
    
    if not left_mirrors or not right_mirrors:
        logger.error("找不到左右臉鏡射圖片")
        return
    
    logger.info(f"找到 {len(left_mirrors)} 張左臉, {len(right_mirrors)} 張右臉")
    
    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置參數
    config = {
        'models': ['vggface', 'arcface', 'dlib', 'deepid'],  # 使用的模型
        'use_topofr': True,
    }
    
    logger.info("\n提取配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 建立特徵提取器
    start_time = datetime.now()
    
    try:
        extractor = FeatureExtractor(
            models=config['models'],
            use_topofr=config['use_topofr'],
        )
        
        logger.info("\n開始提取特徵...")
        
        # 使用 process_folder 方法處理整個資料夾
        results = extractor.process_folder(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            pattern_left="*_Lmirror_claheL.png",
            pattern_right="*_Rmirror_claheL.png"
        )
        
        # 顯示結果統計
        logger.info("\n" + "="*60)
        logger.info("提取結果統計")
        logger.info("="*60)
        
        total_files = sum(len(files) for files in results.values())
        logger.info(f"處理資料夾數: {len(results)}")
        logger.info(f"產生JSON檔案數: {total_files}")
        
        # 顯示每個資料夾的結果
        if results:
            logger.info("\n各資料夾提取結果:")
            for folder, files in results.items():
                logger.info(f"  {folder}: {len(files)} 個JSON檔案")
        
        # 生成總結報告
        if total_files > 0:
            logger.info("\n生成總結報告...")
            summary = extractor.create_summary_report(
                str(output_dir),
                str(output_dir / "extraction_summary.json")
            )
            
            if summary:
                logger.info(f"總共分析了 {summary.get('total_files', 0)} 個檔案")
                
                # 顯示各模型統計
                if 'by_model' in summary:
                    logger.info("\n各模型不對稱性統計:")
                    for model, stats in summary['by_model'].items():
                        logger.info(f"  {model}:")
                        logger.info(f"    平均: {stats['mean_asymmetry']:.4f}")
                        logger.info(f"    標準差: {stats['std_asymmetry']:.4f}")
    
    except Exception as e:
        logger.error(f"提取失敗: {e}", exc_info=True)
        return
    
    # 計算處理時間
    elapsed_time = datetime.now() - start_time
    logger.info(f"\n處理時間: {elapsed_time.total_seconds():.1f} 秒")
    
    # 驗證輸出
    verify_output(output_dir)


def test_single_pair():
    """測試單一影像對（快速測試）"""
    
    input_dir = DATA_PATHS["images"]["preprocessed"]
    output_dir = DATA_PATHS["features"]["datung"] / "test_single"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 找第一對左右臉
    left_files = list(input_dir.rglob("*_Lmirror_claheL.png"))
    right_files = list(input_dir.rglob("*_Rmirror_claheL.png"))
    
    if not left_files or not right_files:
        logger.error("找不到左右臉配對")
        return
    
    # 找出配對
    left_file = left_files[0]
    base_name = left_file.name.replace("_Lmirror_claheL.png", "")
    right_file = None
    
    for rf in right_files:
        if base_name in rf.name:
            right_file = rf
            break
    
    if not right_file:
        logger.error("找不到對應的右臉檔案")
        return
    
    logger.info(f"測試影像對:")
    logger.info(f"  左臉: {left_file.name}")
    logger.info(f"  右臉: {right_file.name}")
    
    # 建立提取器（只用兩個快速的模型）
    extractor = FeatureExtractor(
        models=['vggface', 'arcface', 'dlib', 'deepid'],
        use_topofr=True,
    )
    
    # 處理影像對
    output_path = output_dir / f"{base_name}_LR_difference.json"
    
    result = extractor.process_image_pair(
        str(left_file),
        str(right_file),
        str(output_path)
    )
    
    # 顯示結果
    logger.info("\n提取結果:")
    
    # 顯示各模型成功狀態
    if 'extraction_successful' in result:
        logger.info("模型提取狀態:")
        for model, success in result['extraction_successful'].items():
            status = "✓" if success else "✗"
            logger.info(f"  {model}: {status}")
    
    # 顯示整體不對稱度
    if 'overall_asymmetry' in result:
        overall = result['overall_asymmetry']
        logger.info("\n整體不對稱度:")
        logger.info(f"  平均: {overall['mean_relative_difference']:.4f}")
        logger.info(f"  標準差: {overall['std_relative_difference']:.4f}")
        logger.info(f"  中位數: {overall['median_relative_difference']:.4f}")
    
    # 顯示各模型統計
    if 'statistics' in result:
        logger.info("\n各模型不對稱分數:")
        for model, stats in result['statistics'].items():
            logger.info(f"  {model}: {stats['asymmetry_score']:.4f}")
    
    logger.info(f"\n結果已儲存至: {output_path}")


def test_batch_function():
    """測試批次提取函數"""
    
    input_dir = DATA_PATHS["images"]["preprocessed"]
    output_dir = DATA_PATHS["features"]["datung"] / "test_batch"
    
    logger.info("測試 batch_extract_features 函數...")
    
    # 使用便捷函數
    results = batch_extract_features(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        models=['vggface', 'dlib', 'dlib', 'deepid'],  # 快速測試
        use_topofr=True
    )
    
    total_files = sum(len(files) for files in results.values())
    logger.info(f"批次處理完成: {total_files} 個檔案")


def verify_output(output_dir: Path):
    """驗證輸出結果"""
    logger.info("\n" + "="*60)
    logger.info("驗證輸出")
    logger.info("="*60)
    
    # 收集所有 JSON 檔案
    json_files = list(output_dir.rglob("*_LR_difference.json"))
    
    logger.info(f"找到 {len(json_files)} 個差異JSON檔案")
    
    if not json_files:
        logger.warning("沒有找到任何 JSON 檔案")
        return
    
    # 檢查第一個檔案的結構
    sample_file = json_files[0]
    logger.info(f"\n檢查樣本檔案: {sample_file.name}")
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 檢查必要欄位
        required_keys = [
            "source_images",
            "image_names", 
            "embedding_differences",
            "embedding_averages",
            "relative_differences",
            "embedding_dimensions",
            "extraction_successful"
        ]
        
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            logger.warning(f"缺少欄位: {missing_keys}")
        else:
            logger.info("✓ 所有必要欄位都存在")
        
        # 檢查各模型的特徵
        if "embedding_differences" in data:
            models = list(data["embedding_differences"].keys())
            logger.info(f"\n包含的模型: {', '.join(models)}")
            
            for model in models:
                if data["embedding_differences"][model] is not None:
                    dim = len(data["embedding_differences"][model])
                    logger.info(f"  {model}: {dim} 維特徵向量")
                else:
                    logger.warning(f"  {model}: 無特徵")
        
        # 檢查統計資訊
        if "statistics" in data:
            logger.info("\n統計資訊:")
            for model, stats in data["statistics"].items():
                if "asymmetry_score" in stats:
                    score = stats["asymmetry_score"]
                    logger.info(f"  {model} 不對稱分數: {score:.4f}")
    
    except Exception as e:
        logger.error(f"讀取JSON失敗: {e}")
    
    # 檢查總結報告
    summary_file = output_dir / "extraction_summary.json"
    if summary_file.exists():
        logger.info(f"\n✓ 找到總結報告: {summary_file.name}")
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            logger.info(f"  總檔案數: {summary.get('total_files', 0)}")
            
            if 'high_asymmetry_cases' in summary:
                high_cases = summary['high_asymmetry_cases'][:3]
                if high_cases:
                    logger.info(f"  前3個高度不對稱案例:")
                    for case in high_cases:
                        logger.info(f"    {case['folder']}/{case['file']}: {case['asymmetry_score']:.4f}")
        except Exception as e:
            logger.error(f"讀取總結報告失敗: {e}")


if __name__ == "__main__":
    # 完整測試
    # test_extractor()
    
    # 或單一測試（快速測試用）
    # test_single_pair()  # 測試單一影像對
    test_batch_function()  # 測試批次函數