from enum import Enum

class Datasets(Enum):
    """
    `Datasets` enum class

    Args:
        Enum (`Enum`): The `Dataset` `Enum` class.
    """
    BreastCancer = 'BreastCancer'
    CervicalCancer = 'CervicalCancer'

class ModelName(Enum):
    """
    `ModelName` enum class

    Args:
        Enum (`Enum`): The `ModelName` `Enum` class.
    """
    LogisticRegression = 'LogisticRegression'
    KNeighborsClassifier = 'KNeighborsClassifier'
    SVC = 'SVC'
    KMeans = 'KMeans'
    DecisionTreeClassifier = 'DecisionTreeClassifier'
    RandomForestClassifier = 'RandomForestClassifier'
    GradientBoostingClassifier = 'GradientBoostingClassifier'
    XGBClassifier = 'XGBClassifier'
    CatBoostClassifier = 'CatBoostClassifier'
    MLPClassifier = 'MLPClassifier'
    GaussianNB = 'GaussianNB'
    LinearDiscriminantAnalysis = 'LinearDiscriminantAnalysis'
    QuadraticDiscriminantAnalysis = 'QuadraticDiscriminantAnalysis'

class ModelType(Enum):
    """
    `ModelType` enum class

    Args:
        Enum (`Enum`): The `ModelType` `Enum` class.
    """
    BinaryClassification = 'BinaryClassification'
    MulticlassClassification = 'MulticlassClassification'
    Regression = 'Regression'
    Clustering = 'Clustering'
