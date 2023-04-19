import pandas as pd
from src.enums import ModelType, ModelName
from src.data import Data
from src.instance import Instance
from src.influential_instances_identification import InfluentialInstancesIdentification
from src.cleaning import clean_data
import numpy as np
from sklearn.metrics import fbeta_score, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

class Model: 
    """
    The `Model` class.

    This class is used to create AI models and train them on given datasets. 
    It also provides methods to evaluate and visualize the models.
    """

    def __init__(self, data: Data = Data(), model_name: ModelName = ModelName.LogisticRegression) -> None:
        """ 
        Description:
            The constructor of the `Model` class.

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
        self.params = None
        self.data = data
        # self.data.clean_data()
        self.y_pred = None
        self.y_pred_proba = None
        self.beta = None
        self.dfbeta = None
        self.rmse = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.roc_auc_score = None
        self.confusion_matrix = None
        self.predictions = None
        self.influential_instances_identification = None
        self.influential_instances = None

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
        elif model_name == ModelName.DecisionTreeRegressor:
            from sklearn.tree import DecisionTreeRegressor
            self.model_name = ModelName.DecisionTreeRegressor
            self.model = DecisionTreeRegressor()
        else:
            raise ValueError('Invalid model name.')

    def set_params(self, params: dict) -> None:
        """
        `set_params` function

        Description:
            This function sets the parameters of the model.

        Args:
            params (`dict`): The parameters of the model.
        
        Returns:
            `None`
        """
        self.params = params
        self.model.set_params(**params)

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
        elif self.model_name == ModelName.DecisionTreeRegressor:
            self.type = ModelType.Regression
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
        
    def fit(self, X_train:pd.DataFrame=None, y_train:pd.DataFrame=None) -> None:
        """ 
        `fit` function

        Description:
            The method to train the model.

        Args: 
            `None`
        
        Returns:
            `None`
        """
        if X_train is None:
            X_train = self.data.get_X_train()
        if y_train is None:
            y_train = self.data.get_y_train()
        if self.model_name == ModelName.LogisticRegression:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.KMeans:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.KNeighborsClassifier:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.SVC:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.DecisionTreeClassifier:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.RandomForestClassifier:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.GradientBoostingClassifier:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.XGBClassifier:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.CatBoostClassifier:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.MLPClassifier:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.GaussianNB:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.LinearDiscriminantAnalysis:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.QuadraticDiscriminantAnalysis:
            self.model.fit(X_train, y_train)
        elif self.model_name == ModelName.DecisionTreeRegressor:
            self.model.fit(X_train, y_train)

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
            return self.model.kneighbors_graph()
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

    def calculate_beta(self):
        """
        `calculate_beta` function

        Description:
            This function calculates the beta score of the model.

        Args:
            `None`

        Returns:
            `None`
        """
        if self.predictions is None:
            raise ValueError("The model has not made any predictions yet. Please use the `predict()` function first.")
        
        self.beta = fbeta_score(self.data.get_y_test(), self.predictions, beta = 1)
        self.data.set_dataset_beta(self.beta)

    def get_beta(self) -> float:
        """
        `get_beta` function

        Description:
            This function returns the beta score.

        Args:
            `None`

        Returns:
            `float`: The beta score.
        """
        return self.beta

    def predict(self, input) -> None:
        """
        `predict` function

        Description:
            The method to predict the test data.
        
        Args:
            * input (`Any`): The input data.
        
        Returns:
            `None`
        """
        if self.model_name == ModelName.LogisticRegression:
            self.predictions_proba = self.model.predict_proba(input)
            if self.predictions_proba is None:
                raise ValueError("The model has not made any predictions yet. Please use the `predict()` function first.")
            else: 
                self.predictions = np.argmax(self.predictions_proba, axis = 1)
        elif self.model_name == ModelName.KMeans:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.KNeighborsClassifier:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.SVC:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.DecisionTreeClassifier:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.RandomForestClassifier:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.GradientBoostingClassifier:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.XGBClassifier:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.CatBoostClassifier:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.MLPClassifier:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.GaussianNB:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.LinearDiscriminantAnalysis:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.QuadraticDiscriminantAnalysis:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.LinearRegression:
            self.predictions = self.model.predict(input)
        elif self.model_name == ModelName.DecisionTreeRegressor:
            self.predictions = self.model.predict(input)

    def get_predictions(self) -> (pd.Series | np.ndarray | pd.DataFrame | None):
        """
        `get_prediction` function

        Description:
            This function returns the predictions of the model.
        
        Args:
            `None`
        
        Returns:
            `(pd.Series | np.ndarray | pd.DataFrame | None)`: The predictions of the model.
        """
        return self.predictions

    def get_mse(self, forInstance: False) -> float:
        """
        `get_mse` function

        Description:
            This function calculates and returns the mean squared error of the model.
        
        Args:
            `forInstance` (`bool`): Set this to `True` only if the function is called to calculate the MSE for an instance.
        
        Returns:
            `float`: The mean squared error of the model.
        """
        if self.predictions is None:
            raise ValueError("The model has not made any predictions yet. Please use the `predict()` function first.")
        
        self.mse = mean_squared_error(self.data.get_y_test(), self.predictions)
        if forInstance:
            return self.mse
        self.data.set_dataset_mse(self.mse)
        return self.mse
    
    def get_rmse(self, forInstance: False) -> float:
        """
        `get_rmse` function

        Description:
            This function caclulates and returns the root mean squared error of the model.
        
        Args:
            `forInstance` (`bool`): Set this to `True` only if the function is called to calculate the RMSE for an instance.
        
        Returns:
            `float`: The root mean squared error of the model.
        """
        if self.predictions is None:
            raise ValueError("The model has not made any predictions yet. Please use the `predict()` function first.")
        self.rmse = np.sqrt(mean_squared_error(self.data.get_y_test(), self.predictions))
        if forInstance:
            return self.rmse
        self.data.set_dataset_rmse(self.rmse)
        return self.rmse

    def get_mae(self, forInstance: False) -> float:
        """
        `get_mae` function

        Description:
            This function calculates and returns the mean absolute error of the model.
        
        Args:
            `forInstance` (`bool`): Set this to `True` only if the function is called to calculate the MAE for an instance.
        
        Returns:
            `float`: The mean absolute error of the model.
        """
        if self.predictions is None:
            raise ValueError("The model has not made any predictions yet. Please use the `predict()` function first.")
        self.mae = mean_absolute_error(self.data.get_y_test(), self.predictions)
        if forInstance:
            return self.mae
        self.data.set_dataset_mae(self.mae)
        return self.mae

    def get_accuracy(self, forInstance: False) -> float:
        """
        `get_accuracy` function

        Description:
            This function calculates and returns the accuracy of the model.
        
        Args:
            `self` (`Model`): The model object.
            `forInstance` (`bool`): Set this to `True` only if the function is called to calculate the accuracy for an instance.
        
        Returns:
            `float`: The accuracy of the model.
        """
        self.accuracy = accuracy_score(self.data.get_y_test(), self.predictions)
        if forInstance:
            return self.accuracy
        self.data.set_dataset_accuracy(self.accuracy)
        return self.accuracy
    
    def get_f1_score(self, forInstance: False) -> float:
        """
        `get_f1_score` function

        Description:
            This function calculates and returns the f1 score of the model.
        
        Args:
            `self` (`Model`): The model object.
            `forInstance` (`bool`): Set this to `True` only if the function is called to calculate the f1 score for an instance.
        
        Returns:
            `float`: The f1 score of the model.
        """
        self.f1_score = f1_score(self.data.get_y_test(), self.predictions)
        if forInstance:
            return self.f1_score
        self.data.set_dataset_f1_score(self.f1_score)
        return self.f1_score

    def get_precision(self, forInstance: False) -> float:
        """
        `get_precision` function

        Description:
            This function calculates and returns the precision of the model.
        
        Args:
            `self` (`Model`): The model object.
            `forInstance` (`bool`): Set this to `True` only if the function is called to calculate the precision for an instance.
        
        Returns:
            `float`: The precision of the model.
        """
        self.precision = precision_score(self.data.get_y_test(), self.predictions)
        if forInstance:
            return self.precision
        self.data.set_dataset_precision(self.precision)
        return self.precision
    
    def get_recall(self, forInstance: False) -> float:
        """
        `get_recall` function

        Description:
            This function calculates and returns the recall of the model.
        
        Args:
            `self` (`Model`): The model object.
            `forInstance` (`bool`): Set this to `True` only if the function is called to calculate the recall for an instance.
        
        Returns:
            `float`: The recall of the model.
        """
        self.recall = recall_score(self.data.get_y_test(), self.predictions)
        if forInstance:
            return self.recall
        self.data.set_dataset_recall(self.recall)
        return self.recall

    def train_for_influential_instances(self) -> None:
        """
        `train_for_influential_instances` function

        Description:
            This function trains the dataset without one row each time, and then predicts the test set. 
            
            This is done for each row in the dataset. 
            
            In the case of classification problems, the accuracy, precision, recall and f1 score of each deducted instance is then calculated and stored in the corresponding `Instance` of the `Data` object.
            In the case of regression probels, the MSE, RMSE and MAE of each deducted instance is then calculated and stored in the corresponding `Instance` of the `Data` object.

        Args:
            `None`

        Returns:
            `bool`: `True` if the influential instances were found, `False` otherwise.
        """
        self.data.set_instances()
        if self.type == ModelType.BinaryClassification or self.type == ModelType.MulticlassClassification:
            for i, instance in self.data.get_X_train().iterrows():
                print(f"Training for instance {i}...", end="\r", flush=True)
                X_train = self.data.get_X_train().copy()
                X_train.drop(i, axis=0, inplace=True)
                y_train = self.data.get_y_train().copy()
                y_train.drop(i, axis=0, inplace=True)
                self.fit(X_train=X_train, y_train=y_train)
                self.predict(self.data.get_X_test())
                if self.type == ModelType.BinaryClassification or self.type == ModelType.MulticlassClassification:
                    instance_accuracy = self.get_accuracy(forInstance=True)
                    instance_f1_score = self.get_f1_score(forInstance=True)
                    instance_precision = self.get_precision(forInstance=True)
                    instance_recall = self.get_recall(forInstance=True)
                    instance = Instance(i)
                    instance.set_accuracy(instance_accuracy)
                    instance.set_f1_score(instance_f1_score)
                    instance.set_precision(instance_precision)
                    instance.set_recall(instance_recall)
                    instance.calculate_accuracy_variance(self.data.get_dataset_accuracy())
                    instance.calculate_f1_score_variance(self.data.get_dataset_f1_score())
                    instance.calculate_precision_variance(self.data.get_dataset_precision())
                    instance.calculate_recall_variance(self.data.get_dataset_recall())
                    if instance.is_influential(self.type):
                        self.data.set_instance_as_influential(i)
            influential_instances_found = self.__identify_all_influential_instances()
            return influential_instances_found
        elif self.type == ModelType.Regression:
            for i, instance in self.data.get_X_train().iterrows():
                print(f"Training for instance {i}...", end="\r", flush=True)
                X_train = self.data.get_X_train().copy()
                X_train.drop(i, axis=0, inplace=True)
                y_train = self.data.get_y_train().copy()
                y_train.drop(i, axis=0, inplace=True)
                self.fit(X_train=X_train, y_train=y_train)
                self.predict(self.data.get_X_test())
                instance_mse = self.get_mse(forInstance=True)
                instance_rmse = self.get_rmse(forInstance=True)
                instance_mae = self.get_mae(forInstance=True)
                instance = Instance(i)
                instance.set_mse(instance_mse)
                instance.set_rmse(instance_rmse)
                instance.set_mae(instance_mae)
                instance.calculate_mse_variance(self.data.get_dataset_mse())
                instance.calculate_rmse_variance(self.data.get_dataset_rmse())
                instance.calculate_mae_variance(self.data.get_dataset_mae())
                if instance.is_influential(model_type=self.type):
                    self.data.set_instance_as_influential(i)
            influential_instances_found = self.__identify_all_influential_instances()
            return influential_instances_found
 
    def __identify_all_influential_instances(self) -> None:
        """
        `identify_influential_instances` function

        Description:
            This function identifies the influential instances in the dataset and stores them in the `Data` object.

        Args:
            `None`

        Returns:
            `bool`: `True` if influential instances were found, `False` otherwise.
        """
        self.influential_instances_identification = InfluentialInstancesIdentification(self.data.get_influential_instances(), self.data.get_dataset())
        influential_instances_found = self.influential_instances_identification.identify_influential_instances()
        return influential_instances_found
    
    def fit_with_influential_instances(self):
        """
        `fit_with_influential_instances` function

        Description:
            This function trains the model with the influential instances only.

        Args:
            `None`

        Returns:
            `None`
        """
        print("Training with influential instances...")
        self.influential_instances = self.influential_instances_identification.get_influential_instances()
        self.data.set_X_train(self.influential_instances.drop(self.data.get_y_train().name, axis=1))
        self.data.set_y_train(self.influential_instances[self.data.get_y_train().name])
        self.fit(self.data.get_X_train(), self.data.get_y_train())
