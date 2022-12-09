from enum import Enum

class ModelName(Enum):
    """
    `ModelName` enum class

    Args:
        Enum (`Enum`): The `Enum` class.
    """
    LogisticRegression = 'LogisticRegression'
    KNeighborsClassifier = 'KNeighborsClassifier'
    SVC = 'SVC'
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
    """`ModelType` enum class

    Args:
        Enum (`Enum`): The `Enum` class.
    """
    BinaryClassification = 'BinaryClassification'
    MulticlassClassification = 'MulticlassClassification'
    Regression = 'Regression'
    Clustering = 'Clustering'
