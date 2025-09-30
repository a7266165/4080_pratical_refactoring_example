# scripts/main.py
"""主程式"""
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataloader.dataloader import DataLoader
from src.modeltrainer.modeltrainer import ModelTrainer
from src.reporter.reporter import Reporter

# ==================== 配置設定 ====================
# 特徵配置
EMBEDDING_MODELS = ["vggface", "arcface", "dlib", "deepid", "topofr"]
FEATURE_TYPES = ["difference", "average", "relative"]

# 資料篩選配置
CDR_THRESHOLDS = [0.5, 1.0, 2.0]  # 設為 [] 或 None 停用CDR篩選
USE_ALL_VISITS = True              # False = 只用最新訪視
AGE_MATCHING = True                # False = 停用年齡配對

# 模型訓練配置
MODEL_TYPES = ["random_forest", "xgboost", "svm", "logistic"]
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_SEED = 42

# 輸出配置
OUTPUT_DIR = "./output"

# ==================== 日誌設定 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    """主程式流程"""
    
    # 建立輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(OUTPUT_DIR) / f"run_{timestamp}"
    
    logger.info("=" * 70)
    logger.info("開始執行臉部不對稱分析")
    logger.info("=" * 70)
    
    # 顯示配置
    print_configuration()
    
    # ========== Step 1: 載入資料 ==========
    logger.info("\n[Step 1] 載入資料")
    logger.info("-" * 40)
    
    dataloader = DataLoader(
        embedding_models=EMBEDDING_MODELS,
        feature_types=FEATURE_TYPES,
        use_all_visits=USE_ALL_VISITS,
        age_matching=AGE_MATCHING,
        cdr_thresholds=CDR_THRESHOLDS
    )
    
    datasets = dataloader.load()
    
    logger.info(f"\n成功載入 {len(datasets)} 個資料集配置")
    
    # 顯示資料集統計
    print_dataset_statistics(datasets)
    
    # ========== Step 2: 訓練模型 ==========
    logger.info("\n[Step 2] 訓練模型")
    logger.info("-" * 40)
    
    trainer = ModelTrainer(
        output_dir=str(output_path),
        model_types=MODEL_TYPES,
        random_state=RANDOM_SEED
    )
    
    results = trainer.train_and_save(
        datasets=datasets,
        test_size=TEST_SIZE,
        cv_folds=CV_FOLDS
    )
    
    # ========== Step 3: 生成報告 ==========
    logger.info("\n[Step 3] 生成報告")
    logger.info("-" * 40)
    
    reporter = Reporter(output_dir=str(output_path))
    reporter.generate(results)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"執行完成！結果已儲存至: {output_path}")
    logger.info("=" * 70)
    
    return results


def print_configuration():
    """顯示執行配置"""
    logger.info("\n執行配置:")
    logger.info("-" * 40)
    logger.info(f"嵌入模型: {', '.join(EMBEDDING_MODELS)}")
    logger.info(f"特徵類型: {', '.join(FEATURE_TYPES)}")
    logger.info(f"CDR篩選: {CDR_THRESHOLDS if CDR_THRESHOLDS else '停用'}")
    logger.info(f"使用所有訪視: {'是' if USE_ALL_VISITS else '否（僅最新）'}")
    logger.info(f"年齡配對: {'啟用' if AGE_MATCHING else '停用'}")
    logger.info(f"模型類型: {', '.join(MODEL_TYPES)}")
    logger.info(f"測試集比例: {TEST_SIZE:.1%}")
    logger.info(f"交叉驗證折數: {CV_FOLDS}")


def print_dataset_statistics(datasets: dict):
    """顯示資料集統計"""
    logger.info("\n資料集統計:")
    logger.info("-" * 40)
    
    # 按CDR閾值分組
    by_cdr = {}
    for key, dataset in datasets.items():
        cdr = dataset["metadata"].get("cdr_threshold", "無篩選")
        if cdr not in by_cdr:
            by_cdr[cdr] = {
                "count": 0,
                "total_samples": 0,
                "total_health": 0,
                "total_patient": 0
            }
        
        by_cdr[cdr]["count"] += 1
        by_cdr[cdr]["total_samples"] += dataset["metadata"]["n_samples"]
        by_cdr[cdr]["total_health"] += dataset["metadata"]["n_health"]
        by_cdr[cdr]["total_patient"] += dataset["metadata"]["n_patient"]
    
    for cdr, stats in by_cdr.items():
        cdr_str = f"CDR>{cdr}" if cdr != "無篩選" else "無CDR篩選"
        logger.info(
            f"{cdr_str}: {stats['count']} 個配置, "
            f"共 {stats['total_samples']} 樣本 "
            f"(健康={stats['total_health']}, 病患={stats['total_patient']})"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"執行失敗: {e}", exc_info=True)
        sys.exit(1)