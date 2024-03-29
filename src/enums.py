from enum import Enum

class Datasets(Enum):
    """
    `Datasets` enum class

    Args:
        Enum (`Enum`): The `Dataset` `Enum` class.
    """
    BreastCancer = 'BreastCancer'
    CervicalCancer = 'CervicalCancer'
    XD6 = 'XD6'
    Mifem = 'Mifem'
    Corral = 'Corral'
    WineQuality = 'WineQuality'
    StockMarket = 'StockMarket'
    UsersMobility = 'UsersMobility'
    Services = 'Services'

class ModelName(Enum):
    """
    `ModelName` enum class. An `Enum` class that contains the names of the available models.

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
    LinearRegression = 'LinearRegression'
    DecisionTreeRegressor = 'DecisionTreeRegressor'

class ModelType(Enum):
    """
    `ModelType` enum class. An `Enum` class that contains the types of the available model types.

    Args:
        Enum (`Enum`): The `ModelType` `Enum` class.
    """
    BinaryClassification = 'BinaryClassification'
    MulticlassClassification = 'MulticlassClassification'
    Regression = 'Regression'
    Clustering = 'Clustering'
