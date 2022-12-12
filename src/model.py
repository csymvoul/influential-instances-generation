import pandas as pd
from src.enums import ModelType, ModelName
from src.data import Data


class Model: 
    """
    The `Model` class.

    This class is used to create AI models and train them on given datasets. 
    It also provides methods to evaluate and visualize the models.
    """

    def __init__(self, data: Data = Data(), model_name: ModelName = ModelName.LogisticRegression) -> None:
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
        self.model_name = model_name
        self.model = None
        self.set_model(model_name)
        self.set_type()
        self.data = data
        self.y_pred = None
        self.y_pred_proba = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.roc_auc_score = None
        self.confusion_matrix = None

    def set_model(self, model_name: ModelName) -> None:
        """
        `set_model` function

        Description:
            This function sets the model to be used for training.

        Args:
            model (`ModelName`): The name of the model.

        Raises:
            `ValueError`: If the model name is not valid.
        
        Returns:
            `None`
        """
        if model_name == ModelName.LogisticRegression:
            from sklearn.linear_model import LogisticRegression
            self.model_name = ModelName.LogisticRegression
            self.model = LogisticRegression()
        elif model_name == ModelName.KNeighborsClassifier:
            from sklearn.neighbors import KNeighborsClassifier
            self.model_name = ModelName.KNeighborsClassifier
            self.model = KNeighborsClassifier()
        elif model_name == ModelName.KMeans:
            from sklearn.cluster import KMeans
            self.model_name = ModelName.KMeans
            self.model = KMeans()
        elif model_name == ModelName.SVC:
            from sklearn.svm import SVC
            self.model_name = ModelName.SVC
            self.model = SVC()
        elif model_name == ModelName.DecisionTreeClassifier:
            from sklearn.tree import DecisionTreeClassifier
            self.model_name = ModelName.DecisionTreeClassifier
            self.model = DecisionTreeClassifier()
        elif model_name == ModelName.RandomForestClassifier:
            from sklearn.ensemble import RandomForestClassifier
            self.model_name = ModelName.RandomForestClassifier
            self.model = RandomForestClassifier()
        elif model_name == ModelName.GradientBoostingClassifier:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model_name = ModelName.GradientBoostingClassifier
            self.model = GradientBoostingClassifier()
        elif model_name == ModelName.XGBClassifier:
            from xgboost import XGBClassifier
            self.model_name = ModelName.XGBClassifier
            self.model = XGBClassifier()
        elif model_name ==  ModelName.CatBoostClassifier:
            from catboost import CatBoostClassifier
            self.model_name = ModelName.CatBoostClassifier
            self.model = CatBoostClassifier()
        elif model_name == ModelName.MLPClassifier:
            from sklearn.neural_network import MLPClassifier
            self.model_name = ModelName.MLPClassifier
            self.model = MLPClassifier()
        elif model_name == ModelName.GaussianNB:
            from sklearn.naive_bayes import GaussianNB
            self.model_name = ModelName.GaussianNB
            self.model = GaussianNB()
        elif model_name == ModelName.LinearDiscriminantAnalysis:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.model_name = ModelName.LinearDiscriminantAnalysis
            self.model = LinearDiscriminantAnalysis()
        elif model_name == ModelName.QuadraticDiscriminantAnalysis:
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            self.model_name = ModelName.QuadraticDiscriminantAnalysis
            self.model = QuadraticDiscriminantAnalysis()
        elif model_name == ModelName.LinearRegression:
            from sklearn.linear_model import LinearRegression
            self.model_name = ModelName.LinearRegression
            self.model = LinearRegression()
        else:
            raise ValueError('Invalid model name.')

    def set_type(self) -> None:
        """
        `set_type` function

        Description:
            This function sets the type of the model according to the model.
        
        Args:
            `None`

        Returns:
            `None`
        """
        if self.model_name == ModelName.LogisticRegression:
            self.type = ModelType.BinaryClassification
        elif self.model_name == ModelName.KMeans:
            self.type = ModelType.Clustering
        elif self.model_name == ModelName.LinearRegression:
            self.type = ModelType.Regression
        elif self.model_name == ModelName.KNeighborsClassifier:
            self.type = ModelType.MulticlassClassification
        elif self.model_name == ModelName.SVC:
            self.type = ModelType.MulticlassClassification
        elif self.model_name == ModelName.DecisionTreeClassifier:
            self.type = ModelType.MulticlassClassification
        elif self.model_name == ModelName.RandomForestClassifier:
            self.type = ModelType.MulticlassClassification
        elif self.model_name == ModelName.GradientBoostingClassifier:
            self.type = ModelType.MulticlassClassification
        elif self.model_name == ModelName.XGBClassifier:
            self.type = ModelType.MulticlassClassification
        elif self.model_name ==  ModelName.CatBoostClassifier:
            self.type = ModelType.MulticlassClassification
        elif self.model_name == ModelName.MLPClassifier:
            self.type = ModelType.MulticlassClassification
        elif self.model_name == ModelName.GaussianNB:
            self.type = ModelType.MulticlassClassification
        elif self.model_name == ModelName.LinearDiscriminantAnalysis:
            self.type = ModelType.MulticlassClassification
        elif self.model_name == ModelName.QuadraticDiscriminantAnalysis:
            self.type = ModelType.MulticlassClassification
        else:
            raise ValueError('Invalid model name.')

    def get_model_name(self) -> ModelName:
        """
        `get_model` function

        Description:
            This function returns the model name.

        Args:
            `None`

        Returns:
            `ModelName`: The name of the model.
        """
        return self.model_name
    
    def get_model(self):
        """
        `get_model` function

        Description:
            This function returns the model.

        Args:
            `None`

        Returns:
            `(LogisticRegression | KNeighborsClassifier | KMeans | SVC | DecisionTreeClassifier | RandomForestClassifier 
            | GradientBoostingClassifier | XGBClassifier | CatBoostClassifier | MLPClassifier | GaussianNB 
            | LinearDiscriminantAnalysis | QuadraticDiscriminantAnalysis | LinearRegression | None)`: The model or `None` if the model is not set.
        """
        return self.model
    
    def get_type(self) -> ModelType:
        """
        `get_type` function

        Description:
            This function returns the type of the model.

        Args:
            `None`

        Returns:
            `ModelType`: The type of the model.
        """
        return self.type

    def set_data(self, data: Data) -> None:
        """
        `set_dataset` function

        Description:
            This function sets the dataset used for training of the model.

        Args:
            dataset (`pandas.DataFrame`): The dataset used for training of the model.

        Returns:
            `None`
        """
        self.data = data
    
    def get_data(self) -> Data:
        """
        `get_data` function
        
        Description:
            This function returns the dataset used for training of the model.

        Args:
            `None`

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
        self.data.train_test_split( test_size = test_size, 
                                    random_state = random_state)
        
    def fit(self) -> None:
        """ 
        `fit` function

        Description:
            The method to train the model.

        Args: 
            `None`
        
        Returns:
            `None`
        """
        if self.model_name == ModelName.LogisticRegression:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.KMeans:
            self.model.fit(self.data.get_X_train())
        elif self.model_name == ModelName.KNeighborsClassifier:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.SVC:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.DecisionTreeClassifier:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.RandomForestClassifier:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.GradientBoostingClassifier:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.XGBClassifier:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.CatBoostClassifier:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.MLPClassifier:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.GaussianNB:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.LinearDiscriminantAnalysis:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())
        elif self.model_name == ModelName.QuadraticDiscriminantAnalysis:
            self.model.fit(self.data.get_X_train(), self.data.get_y_train())

    def get_weights(self):
        """
        `get_weights` function

        Description:
            This function returns the weights, means or the coefficients of the model.
        
        Args:
            `None`
        
        Returns:
            `(Any | NDArray | ndarray | NDArray[floating[_64Bit]] | Booster | list | None)`: The weights, means or the coefficients of the model or `None` if the model is not supported.
        """
        if self.model_name == ModelName.LogisticRegression:
            return self.model.coef_
        elif self.model_name == ModelName.KMeans:
            return self.model.means_ 
        elif self.model_name == ModelName.KNeighborsClassifier:
            return self.model.means_
        elif self.model_name == ModelName.SVC:
            return self.model.support_vectors_
        elif self.model_name == ModelName.DecisionTreeClassifier:
            return self.model.tree_
        elif self.model_name == ModelName.RandomForestClassifier:
            return self.model.estimators_
        elif self.model_name == ModelName.GradientBoostingClassifier:
            return self.model.estimators_
        elif self.model_name == ModelName.XGBClassifier:
            return self.model.get_booster()
        elif self.model_name == ModelName.CatBoostClassifier:
            return self.model.get_all_params()
        elif self.model_name == ModelName.MLPClassifier:
            return self.model.coefs_
        elif self.model_name == ModelName.GaussianNB:
            return self.model.theta_
        elif self.model_name == ModelName.LinearDiscriminantAnalysis:
            return self.model.coef_
        elif self.model_name == ModelName.QuadraticDiscriminantAnalysis:
            return self.model.theta_
        