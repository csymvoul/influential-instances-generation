import pandas as pd
from model_type import ModelType, ModelName
from sklearn.model_selection import train_test_split

class Model: 
    """
    Description:
        The `Model` class.

        This class is used to create AI models and train them on given datasets. 
        It also provides methods to evaluate and visualize the models.
    """

    def __init__(self, model_name: ModelName = ModelName.LogisticRegression , type: ModelType = ModelType.BinaryClassification, dataset: pd.DataFrame = None) -> None:
        """ 
        Description:
            The constructor of the class `Model`.

        Args: 
            * model_name (`ModelName`): The name of the model. 
                * Default value is `ModelName.LogisticRegression`. 
                * Other values are `ModelName.KNeighborsClassifier`, `ModelName.SVC`, 
                                    `ModelName.DecisionTreeClassifier`, `ModelName.RandomForestClassifier`, 
                                    `ModelName.GradientBoostingClassifier`, `ModelName.XGBClassifier`,  
                                    `ModelName.CatBoostClassifier`, `ModelName.MLPClassifier`, `ModelName.GaussianNB`, 
                                    `ModelName.LinearDiscriminantAnalysis`, `ModelName.QuadraticDiscriminantAnalysis`.
            * type (`ModelType`): The type of the model. 
                * Default value is `ModelType.BinaryClassification`. 
                * Other values are `ModelType.MulticlassClassification`, `ModelType.Regression`.
            * dataset (`pandas.DataFrame`): The dataset used for training of the model. 
                * Default value is `None`.
        
        Returns:
            `None`
        """
        self.model_name = model_name
        self.type = type
        self.dataset = dataset
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
        self.y_pred = None
        self.y_pred_proba = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.roc_auc_score = None
        self.confusion_matrix = None

    def set_model(self, model: ModelName) -> None:
        """`set_model` function

        Args:
            model (`ModelName`): The name of the model.

        Raises:
            `ValueError`: If the model name is not valid.
        
        Returns:
            `None`
        """
        if model == ModelName.LogisticRegression.value:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
            self.type = ModelType.BinaryClassification
        elif model == ModelName.KNeighborsClassifier.value:
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier()
            self.type = ModelType.Clustering
        elif model == ModelName.SVC.value:
            from sklearn.svm import SVC
            self.model = SVC()
            self.type = ModelType.MulticlassClassification
        elif model == ModelName.DecisionTreeClassifier.value:
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier()
            self.type = ModelType.MulticlassClassification
        elif model == ModelName.RandomForestClassifier.value:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier()
            self.type = ModelType.MulticlassClassification
        elif model == ModelName.GradientBoostingClassifier.value:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier()
            self.type = ModelType.MulticlassClassification
        elif model == ModelName.XGBClassifier.value:
            from xgboost import XGBClassifier
            self.model = XGBClassifier()
            self.type = ModelType.MulticlassClassification
        elif model ==  ModelName.CatBoostClassifier.value:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.type = ModelType.MulticlassClassification
        elif model == ModelName.MLPClassifier.value:
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier()
            self.type = ModelType.MulticlassClassification
        elif model == ModelName.GaussianNB.value:
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB()
            self.type = ModelType.MulticlassClassification
        elif model == ModelName.LinearDiscriminantAnalysis.value:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.model = LinearDiscriminantAnalysis()
            self.type = ModelType.MulticlassClassification
        elif model == ModelName.QuadraticDiscriminantAnalysis.value:
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            self.model = QuadraticDiscriminantAnalysis()
            self.type = ModelType.MulticlassClassification
        else:
            raise ValueError('Invalid model name.')
  
    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """ 
        `train_test_split` function

        Description:
            The method to split the dataset into training and testing sets.

        Args: 
            * test_size (`float`): The size of the testing set. 
                * Default value is `0.2`. 
                * Other values are `0.1`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7`, `0.8`, `0.9`.
            * random_state (`int`): The random state. 
                * Default value is `42`.
        
        Returns:
            `None`
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y, 
                                                                                test_size = test_size, 
                                                                                random_state = random_state)
    
    def train(self) -> None:
        """ 
        `train` function

        Description:
            The method to train the model.

        Args: 
            `self` (`Model`): The instance of the class `Model`.
        
        Returns:
            `None`
        """
        self.model.fit(self.X_train, self.y_train)
