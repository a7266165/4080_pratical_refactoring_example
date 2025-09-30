# scripts/main.py
"""主程式 - 臉部不對稱分析完整管線"""
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import ImagePreprocessor
from src.extractor import FeatureExtractor
from src.loader import DataLoader
from src.trainer import ModelTrainer
from src.reporter import Reporter

# ==================== 配置 ====================
class Config:
    """集中管理所有配置"""
    
    # 路徑配置
    RAW_IMAGE_DIR = "./data/raw_images"
    PREPROCESSED_DIR = "./data/preprocessed"
    FEATURES_DIR = "./data/features"
    OUTPUT_DIR = "./output"
    
    # 前處理配置
    PREPROCESS_STEPS = ['select', 'align', 'mirror', 'clahe']
    DETECTION_CONFIDENCE = 0.8
    CLAHE_CLIP_LIMIT = 2.0
    N_SELECT_IMAGES = 10
    
    # 特徵提取配置
    EMBEDDING_MODELS = ["vggface", "arcface", "dlib", "deepid"]
    FEATURE_TYPES = ["difference", "average", "relative"]
    USE_TOPOFR = False  # 需要額外設定 TopoFR 路徑
    TOPOFR_PATH = None  # 例如: "C:/path/to/TopoFR"
    
    # 資料載入配置
    USE_ALL_VISITS = True
    AGE_MATCHING = True
    CDR_THRESHOLDS = [0.5, 1.0, 2.0]  # 設為 [] 停用 CDR 篩選
    
    # 模型訓練配置
    MODEL_TYPES = ["random_forest", "xgboost", "svm", "logistic"]
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    RANDOM_SEED = 42
    
    # 執行配置
    RUN_STEPS = {
        'preprocess': True,
        'extract': True,
        'train': True,
        'report': True
    }


# ==================== 日誌設定 ====================
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """設定日誌"""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
    
    # 降低一些套件的日誌等級
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


# ==================== 主要流程 ====================
class Pipeline:
    """完整的分析管線"""
    
    def __init__(self, config: Config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.OUTPUT_DIR) / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定日誌
        log_file = self.run_dir / "pipeline.log"
        setup_logging("INFO", str(log_file))
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*70)
        self.logger.info("臉部不對稱分析管線")
        self.logger.info("="*70)
        self._print_config()
    
    def _print_config(self):
        """顯示配置"""
        self.logger.info("\n執行配置:")
        self.logger.info("-"*40)
        
        steps = [k for k, v in self.config.RUN_STEPS.items() if v]
        self.logger.info(f"執行步驟: {', '.join(steps)}")
        
        if self.config.RUN_STEPS['preprocess']:
            self.logger.info(f"前處理步驟: {', '.join(self.config.PREPROCESS_STEPS)}")
            self.logger.info(f"每個案選取: {self.config.N_SELECT_IMAGES} 張")
        
        if self.config.RUN_STEPS['extract']:
            self.logger.info(f"嵌入模型: {', '.join(self.config.EMBEDDING_MODELS)}")
            if self.config.USE_TOPOFR:
                self.logger.info(f"TopoFR: 啟用 ({self.config.TOPOFR_PATH})")
        
        if self.config.RUN_STEPS['train']:
            self.logger.info(f"特徵類型: {', '.join(self.config.FEATURE_TYPES)}")
            self.logger.info(f"分類器: {', '.join(self.config.MODEL_TYPES)}")
            self.logger.info(f"CDR篩選: {self.config.CDR_THRESHOLDS or '停用'}")
            self.logger.info(f"年齡配對: {'啟用' if self.config.AGE_MATCHING else '停用'}")
    
    def run(self):
        """執行完整管線"""
        results = {}
        
        # Step 1: 影像前處理
        if self.config.RUN_STEPS['preprocess']:
            self.logger.info("\n" + "="*60)
            self.logger.info("[Step 1] 影像前處理")
            self.logger.info("-"*40)
            
            preprocessed_dir = self.run_dir / "preprocessed"
            results['preprocess'] = self.run_preprocess(
                self.config.RAW_IMAGE_DIR,
                str(preprocessed_dir)
            )
            
            # 更新路徑供下一步使用
            self.config.PREPROCESSED_DIR = str(preprocessed_dir)
        
        # Step 2: 特徵提取
        if self.config.RUN_STEPS['extract']:
            self.logger.info("\n" + "="*60)
            self.logger.info("[Step 2] 特徵提取")
            self.logger.info("-"*40)
            
            features_dir = self.run_dir / "features"
            results['extract'] = self.run_extraction(
                self.config.PREPROCESSED_DIR,
                str(features_dir)
            )
            
            # 更新路徑供下一步使用
            self.config.FEATURES_DIR = str(features_dir)
        
        # Step 3: 模型訓練
        if self.config.RUN_STEPS['train']:
            self.logger.info("\n" + "="*60)
            self.logger.info("[Step 3] 模型訓練")
            self.logger.info("-"*40)
            
            models_dir = self.run_dir / "models"
            results['train'] = self.run_training(
                self.config.FEATURES_DIR,
                str(models_dir)
            )
        
        # Step 4: 生成報告
        if self.config.RUN_STEPS['report'] and 'train' in results:
            self.logger.info("\n" + "="*60)
            self.logger.info("[Step 4] 生成報告")
            self.logger.info("-"*40)
            
            results['report'] = self.generate_report(
                results['train'],
                str(self.run_dir / "reports")
            )
        
        # 儲存管線結果
        self.save_pipeline_results(results)
        
        self.logger.info("\n" + "="*70)
        self.logger.info(f"執行完成！結果已儲存至: {self.run_dir}")
        self.logger.info("="*70)
        
        return results
    
    def run_preprocess(self, input_dir: str, output_dir: str) -> dict:
        """執行影像前處理"""
        try:
            with ImagePreprocessor(
                detection_confidence=self.config.DETECTION_CONFIDENCE,
                clahe_clip_limit=self.config.CLAHE_CLIP_LIMIT,
                n_select=self.config.N_SELECT_IMAGES
            ) as processor:
                results = processor.process_folder(
                    input_dir,
                    output_dir,
                    steps=self.config.PREPROCESS_STEPS
                )
            
            # 統計
            total_images = sum(len(paths) for paths in results.values())
            self.logger.info(f"前處理完成: {len(results)} 個資料夾, 共 {total_images} 張影像")
            
            return results
            
        except Exception as e:
            self.logger.error(f"前處理失敗: {e}")
            raise
    
    def run_extraction(self, input_dir: str, output_dir: str) -> dict:
        """執行特徵提取"""
        try:
            extractor = FeatureExtractor(
                models=self.config.EMBEDDING_MODELS,
                use_topofr=self.config.USE_TOPOFR,
                topofr_path=self.config.TOPOFR_PATH
            )
            
            results = extractor.process_folder(input_dir, output_dir)
            
            # 生成總結報告
            summary_path = Path(output_dir) / "extraction_summary.json"
            summary = extractor.create_summary_report(output_dir, str(summary_path))
            
            # 統計
            total_pairs = sum(len(paths) for paths in results.values())
            self.logger.info(f"特徵提取完成: {len(results)} 個資料夾, 共 {total_pairs} 對影像")
            
            if summary.get("high_asymmetry_cases"):
                self.logger.info(
                    f"發現 {len(summary['high_asymmetry_cases'])} 個高度不對稱案例"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"特徵提取失敗: {e}")
            raise
    
    def run_training(self, features_dir: str, output_dir: str) -> dict:
        """執行模型訓練"""
        try:
            # 載入資料
            self.logger.info("載入特徵資料...")
            dataloader = DataLoader(
                embedding_models=self.config.EMBEDDING_MODELS,
                feature_types=self.config.FEATURE_TYPES,
                use_all_visits=self.config.USE_ALL_VISITS,
                age_matching=self.config.AGE_MATCHING,
                cdr_thresholds=self.config.CDR_THRESHOLDS
            )
            
            datasets = dataloader.load()
            self.logger.info(f"載入 {len(datasets)} 個資料集配置")
            
            # 訓練模型
            self.logger.info("開始訓練模型...")
            trainer = ModelTrainer(
                output_dir=output_dir,
                model_types=self.config.MODEL_TYPES,
                random_state=self.config.RANDOM_SEED
            )
            
            results = trainer.train_and_save(
                datasets=datasets,
                test_size=self.config.TEST_SIZE,
                cv_folds=self.config.CV_FOLDS
            )
            
            self.logger.info(f"訓練完成: {len(results)} 個模型配置")
            
            return results
            
        except Exception as e:
            self.logger.error(f"模型訓練失敗: {e}")
            raise
    
    def generate_report(self, training_results: dict, output_dir: str) -> dict:
        """生成最終報告"""
        try:
            reporter = Reporter(output_dir=output_dir)
            reporter.generate(training_results)
            
            self.logger.info("報告生成完成")
            return {"status": "completed", "path": output_dir}
            
        except Exception as e:
            self.logger.error(f"報告生成失敗: {e}")
            raise
    
    def save_pipeline_results(self, results: dict):
        """儲存管線執行結果"""
        import json
        from src.utils import save_json
        
        summary = {
            "timestamp": self.timestamp,
            "config": {
                k: v for k, v in vars(self.config).items()
                if not k.startswith('_')
            },
            "steps_executed": list(results.keys()),
            "output_dir": str(self.run_dir)
        }
        
        summary_path = self.run_dir / "pipeline_summary.json"
        save_json(summary, summary_path)
        self.logger.info(f"管線摘要已儲存: {summary_path}")


# ==================== 命令列介面 ====================
def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="臉部不對稱分析管線",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本參數
    parser.add_argument(
        "--config",
        type=str,
        help="配置檔案路徑 (JSON 格式)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="輸出目錄"
    )
    
    # 步驟控制
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["preprocess", "extract", "train", "report", "all"],
        default=["all"],
        help="要執行的步驟"
    )
    
    # 資料路徑
    parser.add_argument(
        "--raw-images",
        type=str,
        help="原始影像目錄"
    )
    
    parser.add_argument(
        "--preprocessed",
        type=str,
        help="已前處理影像目錄（跳過前處理時使用）"
    )
    
    parser.add_argument(
        "--features",
        type=str,
        help="已提取特徵目錄（跳過特徵提取時使用）"
    )
    
    # 模型配置
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["vggface", "arcface", "dlib", "deepid", "topofr"],
        help="要使用的嵌入模型"
    )
    
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=["random_forest", "xgboost", "svm", "logistic"],
        help="要使用的分類器"
    )
    
    # 其他參數
    parser.add_argument(
        "--no-age-matching",
        action="store_true",
        help="停用年齡配對"
    )
    
    parser.add_argument(
        "--cdr-thresholds",
        nargs="+",
        type=float,
        help="CDR 篩選閾值"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細輸出"
    )
    
    return parser.parse_args()


def load_config_from_file(config_path: str, config: Config) -> Config:
    """從檔案載入配置"""
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def main():
    """主程式入口"""
    args = parse_args()
    
    # 建立配置
    config = Config()
    
    # 從檔案載入配置（如果提供）
    if args.config:
        config = load_config_from_file(args.config, config)
    
    # 覆寫命令列參數
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    
    if args.raw_images:
        config.RAW_IMAGE_DIR = args.raw_images
    
    if args.preprocessed:
        config.PREPROCESSED_DIR = args.preprocessed
        config.RUN_STEPS['preprocess'] = False
    
    if args.features:
        config.FEATURES_DIR = args.features
        config.RUN_STEPS['extract'] = False
    
    if args.models:
        config.EMBEDDING_MODELS = args.models
    
    if args.classifiers:
        config.MODEL_TYPES = args.classifiers
    
    if args.no_age_matching:
        config.AGE_MATCHING = False
    
    if args.cdr_thresholds is not None:
        config.CDR_THRESHOLDS = args.cdr_thresholds
    
    if args.seed:
        config.RANDOM_SEED = args.seed
    
    # 設定執行步驟
    if "all" not in args.steps:
        config.RUN_STEPS = {
            'preprocess': 'preprocess' in args.steps,
            'extract': 'extract' in args.steps,
            'train': 'train' in args.steps,
            'report': 'report' in args.steps
        }
    
    # 設定日誌等級
    log_level = "DEBUG" if args.verbose else "INFO"
    
    # 執行管線
    try:
        pipeline = Pipeline(config)
        results = pipeline.run()
        return 0
        
    except KeyboardInterrupt:
        print("\n中斷執行")
        return 1
        
    except Exception as e:
        print(f"\n執行失敗: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())