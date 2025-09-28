# scripts/main.py
"""主程式"""
import logging
from src.data.picture_loader import PictureLoader
from src.featureextractor.featureextractor import FeatureExtractor
from src.featureloader.featureloader import FeatureLoader
from src.modeltrainer.modeltrainer import ModelTrainer
from src.reporter.reporter import Reporter
from src.modeltrainer.trainfactory.classifiers import ClassifierType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("main")

def main():
    """主程式"""
    
    #------------Settings------------#
    EMBEDDING_MODELS = ['vggface', 'arcface', 'dlib', 'deepid']
    FEATURE_TYPES = ['difference', 'average', 'relative']
    USE_ALL_VISITS = True
    AGE_MATCHING = True
    CDR_FILTER = True
    CV_METHODS = ["LOSO", "5-Fold"]
    CLASSIFIERS = [
        ClassifierType.RANDOM_FOREST,
        ClassifierType.XGB,
        ClassifierType.SVM,
        ClassifierType.LOGISTIC
    ]
    
    #------------Main Pipeline------------#
    
    log.info("步驟1: 載入圖片")
    pics = PictureLoader().load()
    if not pics:
        raise RuntimeError("沒有載入到任何圖片")

    log.info("步驟2: 萃取特徵")
    features = FeatureExtractor(EMBEDDING_MODELS, FEATURE_TYPES).extract(pics)
    log.info("features keys=%s", list(features.keys()))

    log.info("步驟3: 載入特徵資料集")
    dataset = FeatureLoader(
        EMBEDDING_MODELS,
        FEATURE_TYPES,
        USE_ALL_VISITS,
        AGE_MATCHING,
        CDR_FILTER
    ).load(features)
    log.info("dataset keys=%s", list(dataset.keys()))

    log.info("步驟4: 訓練模型")
    results = ModelTrainer(CV_METHODS, CLASSIFIERS).train(dataset)
    log.info("步驟5: 生成報告")
    Reporter().generate(results)
    log.info("\n訓練完成！")

if __name__ == "__main__":
    main()