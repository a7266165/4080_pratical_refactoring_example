# scripts/main.py
"""主程式"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import logging
from datetime import datetime
from pathlib import Path

# from src.data.picture_loader import PictureLoader #　TODO: 尚未實作
# from src.featureextractor.featureextractor import FeatureExtractor #　TODO: 尚未實作
from src.dataloader.dataloader import DataLoader
from src.modeltrainer.modeltrainer import ModelTrainer

# 配置設定
EMBEDDING_MODELS = ["vggface", "arcface", "dlib", "deepid", "topofr"]
FEATURE_TYPES = ["difference", "average", "relative"]
CDR_THRESHOLDS = [0.5, 1.0, 2.0]
USE_ALL_VISITS = True
AGE_MATCHING = True

# 模型訓練設定
MODEL_TYPES = ["random_forest", "svm", "logistic", "xgboost"]  # 支援多個模型類型
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_SEED = 42
OUTPUT_DIR = "./output"

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """主程式流程"""

    # 建立輸出目錄（包含時間戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(OUTPUT_DIR) / f"run_{timestamp}"

    logger.info("=" * 70)
    logger.info("開始執行主程式")
    logger.info("=" * 70)

    # 步驟1: 載入資料
    logger.info("步驟1: 初始化資料載入器")
    dataloader = DataLoader(
        embedding_models=EMBEDDING_MODELS,
        feature_types=FEATURE_TYPES,
        use_all_visits=USE_ALL_VISITS,
        age_matching=AGE_MATCHING,
        cdr_thresholds=CDR_THRESHOLDS,
    )

    logger.info("步驟2: 載入特徵資料")
    datasets = dataloader.load()

    logger.info(f"步驟3: 成功載入資料集")
    logger.info(f"dataset keys={list(datasets.keys())}")

    # 步驟2: 訓練模型
    logger.info("步驟4: 訓練模型")
    trainer = ModelTrainer(
        output_dir=str(output_path),
        model_types=MODEL_TYPES,  # 注意：改成 model_types (複數)
        random_state=RANDOM_SEED,
    )

    results = trainer.train_and_save(
        datasets=datasets, test_size=TEST_SIZE, cv_folds=CV_FOLDS
    )

    # 步驟3: 顯示結果
    logger.info("步驟5: 訓練完成，顯示結果")
    display_results(results)

    logger.info(f"所有模型已儲存至: {output_path}")
    logger.info("=" * 70)

    return results


def display_results(results):
    """顯示訓練結果摘要"""
    logger.info("\n訓練結果摘要:")
    logger.info("-" * 50)

    # 找出最佳配置（基於 MCC）
    best_mcc = -1  # MCC 範圍是 [-1, 1]
    best_accuracy = 0
    best_config = None
    best_config_acc = None

    # 按模型類型分組顯示結果
    model_types = set()
    for config_name in results.keys():
        # 提取模型類型（配置名稱的最後一部分）
        model_type = config_name.split("_")[-1]
        model_types.add(model_type)

    for model_type in sorted(model_types):
        logger.info(f"\n{model_type.upper()} 模型結果:")
        logger.info("-" * 30)

        for config_name, result in results.items():
            if config_name.endswith(f"_{model_type}"):
                test_acc = result["test_metrics"]["accuracy"]
                test_mcc = result["test_metrics"]["mcc"]
                cv_mean = result["cv_mean"]

                # 簡化配置名稱的顯示
                display_name = config_name.replace(f"_{model_type}", "")
                logger.info(f"  {display_name}:")
                logger.info(f"    測試準確率: {test_acc:.4f}")
                logger.info(f"    測試 MCC: {test_mcc:.4f}")
                logger.info(f"    交叉驗證: {cv_mean:.4f} (+/- {result['cv_std']:.4f})")

                # 追蹤最佳 MCC
                if test_mcc > best_mcc:
                    best_mcc = test_mcc
                    best_config = config_name

                # 也追蹤最佳準確率
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_config_acc = config_name

    # 顯示最佳結果
    logger.info("\n" + "=" * 50)
    logger.info("最佳模型結果:")
    logger.info("=" * 50)

    # 基於 MCC 的最佳模型
    if best_config:
        best_result = results[best_config]
        logger.info(f"\n基於 MCC 的最佳配置: {best_config}")
        logger.info(f"  測試 MCC: {best_result['test_metrics']['mcc']:.4f}")
        logger.info(f"  測試準確率: {best_result['test_metrics']['accuracy']:.4f}")
        logger.info(f"  測試 F1: {best_result['test_metrics']['f1']:.4f}")
        logger.info(f"  測試精確率: {best_result['test_metrics']['precision']:.4f}")
        logger.info(f"  測試召回率: {best_result['test_metrics']['recall']:.4f}")
        if "auc" in best_result["test_metrics"]:
            logger.info(f"  測試 AUC: {best_result['test_metrics']['auc']:.4f}")
        logger.info(
            f"  交叉驗證: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']:.4f})"
        )

    # 如果基於準確率的最佳模型不同，也顯示
    if best_config_acc and best_config_acc != best_config:
        logger.info(f"\n基於準確率的最佳配置: {best_config_acc}")
        best_result_acc = results[best_config_acc]
        logger.info(f"  測試準確率: {best_result_acc['test_metrics']['accuracy']:.4f}")
        logger.info(f"  測試 MCC: {best_result_acc['test_metrics']['mcc']:.4f}")

    logger.info("=" * 50)


if __name__ == "__main__":
    main()
