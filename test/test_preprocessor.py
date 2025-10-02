# test_preprocessor.py
"""測試影像前處理模組"""
import sys
from pathlib import Path
import logging
from datetime import datetime

# 設定路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.path_config import DATA_PATHS
from src.preprocessor import ImagePreprocessor

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_preprocessor():
    """測試前處理器"""
    
    # 路徑設定
    input_dir = DATA_PATHS["images"]["raw"]
    output_dir = DATA_PATHS["images"]["preprocessed"]
    
    logger.info("="*60)
    logger.info("開始測試影像前處理器")
    logger.info("="*60)
    logger.info(f"輸入路徑: {input_dir}")
    logger.info(f"輸出路徑: {output_dir}")
    
    # 檢查輸入路徑是否存在
    if not input_dir.exists():
        logger.error(f"輸入路徑不存在: {input_dir}")
        return
    
    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置參數
    config = {
        'n_select': 10,                    # 每個資料夾選擇10張最佳照片
        'detection_confidence': 0.8,       # MediaPipe 偵測信心度
        'clahe_clip_limit': 2.0,          # CLAHE 參數
        'clahe_tile_size': 8,             # CLAHE tile 大小
        'mirror_size': (512, 512),        # 輸出影像大小
        'feather_px': 2                   # 邊緣羽化像素
    }
    
    logger.info("\n處理配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 建立前處理器並執行
    start_time = datetime.now()
    
    try:
        with ImagePreprocessor(**config) as preprocessor:
            logger.info("\n開始處理...")
            
            # 可選擇要執行的步驟
            steps = ['select', 'mirror', 'clahe', 'align']  # 完整流程
            # steps = ['select', 'mirror', 'clahe']  # 不做角度校正
            # steps = ['mirror', 'clahe', 'align']   # 不做篩選
            
            logger.info(f"執行步驟: {' -> '.join(steps)}")
            
            # 處理資料夾
            results = preprocessor.process_folder(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                steps=steps
            )
            
            # 顯示結果統計
            logger.info("\n" + "="*60)
            logger.info("處理結果統計")
            logger.info("="*60)
            
            total_files = sum(len(files) for files in results.values())
            logger.info(f"處理資料夾數: {len(results)}")
            logger.info(f"產生檔案總數: {total_files}")
            
            # 顯示每個資料夾的結果
            if results:
                logger.info("\n各資料夾處理結果:")
                for folder, files in results.items():
                    logger.info(f"  {folder}: {len(files)} 個檔案")
                    
                    # 顯示前3個檔案名稱作為範例
                    if files:
                        logger.info("    範例檔案:")
                        for file in files[:3]:
                            logger.info(f"      - {Path(file).name}")
                        if len(files) > 3:
                            logger.info(f"      ... (還有 {len(files)-3} 個)")
    
    except Exception as e:
        logger.error(f"處理失敗: {e}", exc_info=True)
        return
    
    # 計算處理時間
    elapsed_time = datetime.now() - start_time
    logger.info(f"\n處理時間: {elapsed_time.total_seconds():.1f} 秒")
    
    # 驗證輸出
    verify_output(output_dir)


def verify_output(output_dir: Path):
    """驗證輸出結果"""
    logger.info("\n" + "="*60)
    logger.info("驗證輸出")
    logger.info("="*60)
    
    # 統計各類檔案
    left_mirrors = list(output_dir.rglob("*_Lmirror_claheL.png"))
    right_mirrors = list(output_dir.rglob("*_Rmirror_claheL.png"))
    other_files = list(output_dir.rglob("*_processed.png"))
    
    logger.info(f"左臉鏡射: {len(left_mirrors)} 個")
    logger.info(f"右臉鏡射: {len(right_mirrors)} 個")
    logger.info(f"其他處理: {len(other_files)} 個")
    
    # 檢查配對
    if left_mirrors and right_mirrors:
        logger.info("\n檢查左右臉配對...")
        
        left_stems = {p.stem.replace("_Lmirror_claheL", "") for p in left_mirrors}
        right_stems = {p.stem.replace("_Rmirror_claheL", "") for p in right_mirrors}
        
        matched = left_stems & right_stems
        left_only = left_stems - right_stems
        right_only = right_stems - left_stems
        
        logger.info(f"  配對成功: {len(matched)} 對")
        if left_only:
            logger.warning(f"  只有左臉: {len(left_only)} 個")
        if right_only:
            logger.warning(f"  只有右臉: {len(right_only)} 個")
    
    # 檢查資料夾結構
    logger.info("\n輸出資料夾結構:")
    subdirs = [d for d in output_dir.rglob("*") if d.is_dir()]
    for subdir in sorted(subdirs)[:10]:  # 顯示前10個
        rel_path = subdir.relative_to(output_dir)
        file_count = len(list(subdir.glob("*.png")))
        if file_count > 0:
            logger.info(f"  {rel_path}: {file_count} 個檔案")
    
    if len(subdirs) > 10:
        logger.info(f"  ... (還有 {len(subdirs)-10} 個資料夾)")


def test_single_folder(folder_name: str = None):
    """測試單一資料夾（用於快速測試）"""
    
    input_dir = DATA_PATHS["images"]["raw"]
    output_dir = DATA_PATHS["images"]["preprocessed"] / "test_single"
    
    if folder_name:
        # 找特定資料夾
        target_folder = None
        for path in input_dir.rglob(folder_name):
            if path.is_dir():
                target_folder = path
                break
        
        if not target_folder:
            logger.error(f"找不到資料夾: {folder_name}")
            return
        
        input_dir = target_folder
    else:
        # 找第一個有圖片的資料夾
        for path in input_dir.rglob("*"):
            if path.is_dir() and list(path.glob("*.jpg")):
                input_dir = path
                break
    
    logger.info(f"測試單一資料夾: {input_dir}")
    
    with ImagePreprocessor(n_select=10) as preprocessor:
        results = preprocessor.process_folder(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            steps=['select', 'mirror', 'clahe', 'align']
        )
        
        logger.info(f"處理完成: {sum(len(f) for f in results.values())} 個檔案")


if __name__ == "__main__":
    # 完整測試
    test_preprocessor()
    
    # 或測試單一資料夾（快速測試用）
    # test_single_folder("ACS1-1")  # 指定資料夾
    # test_single_folder()  # 自動選擇第一個