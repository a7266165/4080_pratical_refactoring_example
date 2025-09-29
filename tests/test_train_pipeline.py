# tests/test_train_pipeline.py
"""測試訓練管線（步驟3-5）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import traceback
import numpy as np

# 使用新的模組，不再依賴 legacy_V2
from src.dataloader.dataloader import DataLoader
from src.modeltrainer.modeltrainer import ModelTrainer
from src.modeltrainer.trainfactory.classifiers import ClassifierType
from src.reporter.reporter import Reporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("test")


def test_feature_loader():
    """測試特徵載入器"""
    log.info("\n" + "=" * 60)
    log.info("測試 DataLoader")
    log.info("=" * 60)

    try:

        # 測試配置：先用最小集合測試
        EMBEDDING_MODELS = ["vggface"]  # 先測試一個
        FEATURE_TYPES = ["difference"]  # 先測試一個
        USE_ALL_VISITS = False
        AGE_MATCHING = False  # 先關閉，避免複雜度
        CDR_FILTER = False  # 先關閉

        loader = DataLoader(
            EMBEDDING_MODELS, FEATURE_TYPES, USE_ALL_VISITS, AGE_MATCHING, CDR_FILTER
        )

        # 載入特徵
        dataset = loader.load()

        log.info(f"成功載入資料集")
        log.info(f"  Keys: {list(dataset.keys())}")

        # 檢查資料格式
        for key, data in dataset.items():
            log.info(f"\n  {key}:")
            log.info(f"    X shape: {data['X'].shape}")
            log.info(f"    y shape: {data['y'].shape}")
            log.info(f"    健康: {np.sum(data['y']==0)}, 病患: {np.sum(data['y']==1)}")
            log.info(f"    受試者數: {len(set(data['subject_ids']))}")

        return dataset

    except Exception as e:
        log.error(f"FeatureLoader 測試失敗: {e}")
        traceback.print_exc()
        return None


def test_model_trainer(dataset):
    """測試模型訓練器"""
    log.info("\n" + "=" * 60)
    log.info("測試 ModelTrainer")
    log.info("=" * 60)

    if not dataset:
        log.warning("跳過（需要先成功載入資料）")
        return None

    try:

        # 測試配置：用較快的配置
        CV_METHODS = ["5-Fold"]  # LOSO較慢，先用5-Fold
        CLASSIFIERS = [
            ClassifierType.RANDOM_FOREST,  # 最快的
            # ClassifierType.XGB,  # 如果要測試更多
        ]

        trainer = ModelTrainer(CV_METHODS, CLASSIFIERS)

        # 為了加快測試，可以只用部分資料
        test_dataset = {}
        for key, data in dataset.items():
            # 取前10000筆資料做測試
            n_samples = min(10000, len(data["X"]))
            test_dataset[key] = {
                "X": data["X"][:n_samples],
                "y": data["y"][:n_samples],
                "subject_ids": data["subject_ids"][:n_samples],
                "metadata": data.get("metadata", {}),
            }
            log.info(f"使用 {n_samples} 筆資料測試 {key}")

        # 訓練模型
        results = trainer.train(test_dataset)

        log.info("訓練完成")
        log.info(f"  結果結構: {list(results.keys())}")

        # 顯示部分結果
        for dataset_name, dataset_results in results.items():
            log.info(f"\n  {dataset_name}:")
            for cv_method, cv_results in dataset_results.items():
                log.info(f"    {cv_method}:")
                for clf_name, metrics in cv_results.items():
                    if "error" not in metrics:
                        log.info(
                            f"      {clf_name}: Acc={metrics['accuracy']:.3f}, MCC={metrics['mcc']:.3f}"
                        )

        return results

    except Exception as e:
        log.error(f"ModelTrainer 測試失敗: {e}")
        traceback.print_exc()
        return None


def test_reporter(results):
    """測試報告生成器"""
    log.info("\n" + "=" * 60)
    log.info("測試 Reporter")
    log.info("=" * 60)

    if not results:
        log.warning("跳過（需要先成功訓練）")
        return

    try:

        # 使用測試輸出目錄
        reporter = Reporter(output_dir="test_results")

        # 生成報告
        reporter.generate(results)

        log.info("報告生成成功")

    except Exception as e:
        log.error(f"Reporter 測試失敗: {e}")
        traceback.print_exc()


def test_full_pipeline():
    """測試完整流程（步驟3-5）"""
    log.info("\n" + "#" * 60)
    log.info("# 測試訓練管線（步驟3-5）")
    log.info("#" * 60)

    # 步驟3: 載入特徵
    log.info("\n步驟3: 載入特徵資料集")
    dataset = test_feature_loader()

    if dataset:
        log.info("✓ FeatureLoader 測試通過")
    else:
        log.error("✗ FeatureLoader 測試失敗")
        return

    # 步驟4: 訓練模型
    log.info("\n步驟4: 訓練模型")
    results = test_model_trainer(dataset)

    if results:
        log.info("✓ ModelTrainer 測試通過")
    else:
        log.error("✗ ModelTrainer 測試失敗")
        return

    # 步驟5: 生成報告
    log.info("\n步驟5: 生成報告")
    test_reporter(results)

    log.info("\n" + "#" * 60)
    log.info("# 測試完成！")
    log.info("#" * 60)


def quick_test():
    """快速測試（使用更完整的配置）"""
    log.info("\n執行快速完整測試...")

    try:
        # 實際要用的配置
        EMBEDDING_MODELS = ["vggface", "arcface"]  # 測試兩個模型
        FEATURE_TYPES = ["difference", "average"]  # 測試兩個特徵
        USE_ALL_VISITS = True
        AGE_MATCHING = True
        CDR_FILTER = False  # CDR篩選可能會減少太多資料
        CV_METHODS = ["5-Fold", "LOSO"]
        CLASSIFIERS = [
            ClassifierType.RANDOM_FOREST,
            ClassifierType.XGB,
            ClassifierType.LOGISTIC,
        ]

        log.info("配置：")
        log.info(f"  嵌入模型: {EMBEDDING_MODELS}")
        log.info(f"  特徵類型: {FEATURE_TYPES}")
        log.info(f"  使用所有訪視: {USE_ALL_VISITS}")
        log.info(f"  年齡配對: {AGE_MATCHING}")
        log.info(f"  CDR篩選: {CDR_FILTER}")
        log.info(f"  CV方法: {CV_METHODS}")
        log.info(f"  分類器: {[c.value for c in CLASSIFIERS]}")

        # 執行管線
        log.info("\n步驟3: 載入特徵資料集")
        dataset = DataLoader(
            EMBEDDING_MODELS, FEATURE_TYPES, USE_ALL_VISITS, AGE_MATCHING, CDR_FILTER
        )
        log.info(f"dataset keys={list(dataset.keys())}")

        log.info("\n步驟4: 訓練模型")
        results = ModelTrainer(CV_METHODS, CLASSIFIERS).train(dataset)

        log.info("\n步驟5: 生成報告")
        Reporter().generate(results)

        log.info("\n訓練完成！")

    except Exception as e:
        log.error(f"測試失敗: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="測試訓練管線")
    parser.add_argument("--quick", action="store_true", help="執行快速完整測試")
    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        test_full_pipeline()