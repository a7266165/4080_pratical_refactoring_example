from enum import Enum

class ClassifierType(Enum):
    """支援的分類器類型"""

    RANDOM_FOREST = "Random Forest"
    XGB = "XGBoost"
    SVM = "SVM"
    LOGISTIC = "Logistic Regression"
