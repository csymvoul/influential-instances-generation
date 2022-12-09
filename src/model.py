import pandas as pd
from src.models_enum import ModelType, ModelName
from src.data import Data
from sklearn.model_selection import train_test_split

class Model: 
    """
    Description:
        The `Model` class.

        This class is used to create AI models and train them on given datasets. 
        It also provides methods to evaluate and visualize the models.
    """

    def __init__(self, data: Data = Data(), model: ModelName = ModelName.LogisticRegression ) -> None:
        """ 
        Description:
            The constructor of the class `Model`.

        Args: 
            * model_name (`ModelName`): The name of the model. 
                * Default value is `ModelName.LogisticRegression`. 
                * Other values are `ModelName.KNeighborsClassifier`, `ModelName.SVC`, `ModelName.KMeans`,
                                    `ModelName.DecisionTreeClassifier`, `ModelName.RandomForestClassifier`, 
                                    `ModelName.GradientBoostingClassifier`, `ModelName.XGBClassifier`,  
                                    `ModelName.CatBoostClassifier`, `ModelName.MLPClassifier`, `ModelName.GaussianNB`, 
                                    `ModelName.LinearDiscriminantAnalysis`, `ModelName.QuadraticDiscriminantAnalysis`.
            * dataset (`Data`): The dataset used for training of the model. 
                * Default value is the default `Data`.
        
        Returns:
            `None`
        """
        self.model = model
        self.set_type()
        self.data = data
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
        elif model == ModelName.KNeighborsClassifier.value:
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier()
        elif model == ModelName.KMeans.value:
            from sklearn.cluster import KMeans
            self.model = KMeans()
        elif model == ModelName.SVC.value:
            from sklearn.svm import SVC
            self.model = SVC()
        elif model == ModelName.DecisionTreeClassifier.value:
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier()
        elif model == ModelName.RandomForestClassifier.value:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier()
        elif model == ModelName.GradientBoostingClassifier.value:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier()
        elif model == ModelName.XGBClassifier.value:
            from xgboost import XGBClassifier
            self.model = XGBClassifier()
        elif model ==  ModelName.CatBoostClassifier.value:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
        elif model == ModelName.MLPClassifier.value:
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier()
        elif model == ModelName.GaussianNB.value:
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB()
        elif model == ModelName.LinearDiscriminantAnalysis.value:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.model = LinearDiscriminantAnalysis()
        elif model == ModelName.QuadraticDiscriminantAnalysis.value:
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            self.model = QuadraticDiscriminantAnalysis()
        else:
            raise ValueError('Invalid model name.')

    def set_type(self) -> None:
        """`set_type` function
        Description:
            This function sets the type of the model according to the model.
        
        Args:
            `None`

        Returns:
            `None`
        """

        if self.model.value == ModelName.LogisticRegression.value:
            self.type = ModelType.BinaryClassification
        elif self.model.value == ModelName.KMeans.value:
            self.type = ModelType.Clustering
        elif self.model.value == ModelName.KNeighborsClassifier.value:
            self.type = ModelType.Clustering
        elif self.model.value == ModelName.SVC.value:
            self.type = ModelType.MulticlassClassification
        elif self.model.value == ModelName.DecisionTreeClassifier.value:
            self.type = ModelType.MulticlassClassification
        elif self.model.value == ModelName.RandomForestClassifier.value:
            self.type = ModelType.MulticlassClassification
        elif self.model.value == ModelName.GradientBoostingClassifier.value:
            self.type = ModelType.MulticlassClassification
        elif self.model.value == ModelName.XGBClassifier.value:
            self.type = ModelType.MulticlassClassification
        elif self.model.value ==  ModelName.CatBoostClassifier.value:
            self.type = ModelType.MulticlassClassification
        elif self.model.value == ModelName.MLPClassifier.value:
            self.type = ModelType.MulticlassClassification
        elif self.model.value == ModelName.GaussianNB.value:
            self.type = ModelType.MulticlassClassification
        elif self.model.value == ModelName.LinearDiscriminantAnalysis.value:
            self.type = ModelType.MulticlassClassification
        elif self.model.value == ModelName.QuadraticDiscriminantAnalysis.value:
            self.type = ModelType.MulticlassClassification
        else:
            raise ValueError('Invalid model name.')

    def get_model(self) -> ModelName:
        """`get_model` function

        Returns:
            `ModelName`: The name of the model.
        """
        return self.model
    
    def get_type(self) -> ModelType:
        """`get_type` function

        Returns:
            `ModelType`: The type of the model.
        """
        return self.type

    def set_data(self, data: Data) -> None:
        """`set_dataset` function

        Args:
            dataset (`pandas.DataFrame`): The dataset used for training of the model.

        Returns:
            `None`
        """
        self.data = data
    
    def get_data(self) -> Data:
        """`get_data` function

        Returns:
            `Data`: The dataset used for training of the model.
        """
        return self.data
    
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
