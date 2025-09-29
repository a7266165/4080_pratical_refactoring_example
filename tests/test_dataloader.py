import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataloader.dataloader import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("main")

def main():

    EMBEDDING_MODELS = ["vggface"]
    FEATURE_TYPES = ["difference"]
    USE_ALL_VISITS = False
    AGE_MATCHING = False
    CDR_FILTER = False 

    log.info("步驟3: 載入特徵資料集")
    dataset = DataLoader(
        EMBEDDING_MODELS,
        FEATURE_TYPES,
        USE_ALL_VISITS,
        AGE_MATCHING,
        CDR_FILTER
    ).load()
    log.info("dataset keys=%s", list(dataset.keys()))

if __name__ == "__main__":
    main()