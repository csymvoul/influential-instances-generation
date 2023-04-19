from sklearn.metrics import mean_squared_error, r2_score
from src.enums import ModelType
import pandas as pd
import numpy as np

class Instance():
    """
    The `Instance` class.

    Description:
        This class is used to create an instance of the `Instance` class.
    """

    def __init__(self, instance_index: int) -> None:
        """ 
        Description:
            The constructor of the `Instance` class.

        Args: 
            * `self` (`Instance`): The instance of the class `Instance`.
            * `instance_index` (`int`): The instance's index.
        
        Returns:
            `None`
        """
        self.instance_index = instance_index
        self.instance_score = None
        self.beta = None
        self.rmse = None
        self.r2 = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.dataset_accuracy = None
        self.dataset_precision = None
        self.dataset_recall = None
        self.dataset_f1_score = None
        self.accuracy_variance = None
        self.precision_variance = None
        self.recall_variance = None
        self.f1_score_variance = None
        self.mse_variance = None
        self.rmse_variance = None
        self.mae_variance = None
        self.weights = None
        self.overall_beta = None
        self.dfbeta = None
        self.influential = False
    
    def get_instance_index(self) -> int:
        """
        `get_instance_index` function

        Description:
            This function returns the instance index.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            instance_index (`int`): The instance's index.
        """
        return self.instance_index

    def calculate_dfbeta(self) -> None:
        """
        `calculate_dfbeta` function

        Description:
            This function calculates the dfbeta value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            `None`
        """
        self.dfbeta = self.beta - self.overall_beta
    
    def get_dfbeta(self) -> float:
        """
        `get_dfbeta` function

        Description:
            This function returns the dfbeta value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            dfbeta (`float`): The dfbeta value for each instance in the dataset.
        """
        return self.dfbeta

    def set_overall_beta(self, overall_beta: float) -> None:
        """
        `set_overall_beta` function

        Description:
            This function sets the overall beta value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            overall_beta (`float`): The overall beta value for each instance in the dataset.

        Returns:
            `None`
        """
        self.overall_beta = overall_beta

    def calculate_beta(self) -> None:
        """
        `calculate_beta` function

        Description:
            This function calculates the beta value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            `None`
        """
        pass

    def calculate_rmse(self) -> None:
        """
        `calculate_rmse` function

        Description:
            This function calculates the rmse value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            `None`
        """
        pass

    def calculate_r2(self) -> None:
        """
        `calculate_r2` function

        Description:
            This function calculates the r2 value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            `None`
        """
        pass

    def get_beta(self) -> float: 
        """
        `get_beta` function

        Description:
            This function returns the beta value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            beta (`float`): The beta value for each instance in the dataset.
        """
        return self.beta
    
    def set_mse(self, mse) -> None:
        """
        `set_mse` function

        Description:
            This function sets the MSE value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            mse (`float`): The MSE value for each instance in the dataset.

        Returns:
            `None`
        """
        self.mse = mse
    
    def set_rmse(self, rmse) -> None:
        """
        `set_rmse` function

        Description:
            This function sets the rmse value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            rmse (`float`): The RMSE value for each instance in the dataset.

        Returns:
            `None`
        """
        self.rmse = rmse
    
    def set_mae(self, mae) -> None:
        """
        `set_mae` function

        Description:
            This function sets the MAE value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            mae (`float`): The MAE value for each instance in the dataset.

        Returns:
            `None`
        """
        self.mae = mae

    def get_rmse(self) -> float:
        """
        `get_rmse` function

        Description:
            This function returns the rmse value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            rmse (`float`): The rmse value for each instance in the dataset.
        """
        return self.rmse
    
    def get_r2(self) -> float:
        """
        `get_r2` function

        Description:
            This function returns the r2 value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            r2 (`float`): The r2 value for each instance in the dataset.
        """
        return self.r2

    def get_weights(self) -> float:
        """
        `get_weights` function

        Description:
            This function returns the weights value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            weights (`float`): The weights value for each instance in the dataset.
        """
        return self.weights

    def set_accuracy(self, accuracy) -> None:
        """
        `set_accuracy` function

        Description:
            This function sets the accuracy value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `accuracy` (`float`): The accuracy value for each instance in the dataset.

        Returns:
            `None`
        """
        self.accuracy = accuracy
    
    def set_precision(self, precision) -> None:
        """
        `set_precision` function

        Description:
            This function sets the precision value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `precision` (`float`): The precision value for each instance in the dataset.

        Returns:
            `None`
        """
        self.precision = precision
    
    def set_recall(self, recall) -> None:
        """
        `set_recall` function

        Description:
            This function sets the recall value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `recall` (`float`): The recall value for each instance in the dataset.

        Returns:
            `None`
        """
        self.recall = recall
    
    def set_f1_score(self, f1_score) -> None:
        """
        `set_f1_score` function

        Description:
            This function sets the f1_score value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `f1_score` (`float`): The f1_score value for each instance in the dataset.

        Returns:
            `None`
        """
        self.f1_score = f1_score
    
    def calculate_accuracy_variance(self, dataset_accuracy: float) -> None:
        """
        `calculate_accuracy_variance` function

        Description:
            This function calculates the accuracy variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `dataset_accuracy` (`float`): The accuracy value for the dataset.

        Returns:
            `None`
        """
        self.dataset_accuracy = dataset_accuracy
        self.accuracy_variance = self.accuracy - dataset_accuracy
    
    def calculate_precision_variance(self, dataset_precision: float) -> None:
        """
        `calculate_precision_variance` function

        Description:
            This function calculates the precision variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `dataset_precision` (`float`): The precision value for the dataset.

        Returns:
            `None`
        """
        self.dataset_precision = dataset_precision
        self.precision_variance = self.precision - dataset_precision
    
    def calculate_recall_variance(self, dataset_recall: float) -> None:
        """
        `calculate_recall_variance` function

        Description:
            This function calculates the recall variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `dataset_recall` (`float`): The recall value for the dataset.

        Returns:
            `None`
        """
        self.dataset_recall = dataset_recall
        self.recall_variance = self.recall - dataset_recall
    
    def calculate_f1_score_variance(self, dataset_f1_score: float) -> None:
        """
        `calculate_f1_score_variance` function

        Description:
            This function calculates the f1_score variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `dataset_f1_score` (`float`): The f1_score value for the dataset.

        Returns:
            `None`
        """
        self.dataset_f1_score = dataset_f1_score
        self.f1_score_variance = self.f1_score - dataset_f1_score
    
    def calculate_rmse_variance(self, dataset_rmse: float) -> None:
        """
        `calculate_rmse_variance` function

        Description:
            This function calculates the rmse variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `dataset_rmse` (`float`): The rmse value for the dataset.

        Returns:
            `None`
        """
        self.dataset_rmse = dataset_rmse
        self.rmse_variance = self.rmse - dataset_rmse
    
    def calculate_mse_variance(self, dataset_mse: float) -> None:
        """
        `calculate_mse_variance` function

        Description:
            This function calculates the mse variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `dataset_mse` (`float`): The mse value for the dataset.

        Returns:
            `None`
        """
        self.dataset_mse = dataset_mse
        self.mse_variance = self.mse - dataset_mse
    
    def calculate_mae_variance(self, dataset_mae: float) -> None:
        """
        `calculate_mae_variance` function

        Description:
            This function calculates the mae variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `dataset_mae` (`float`): The mae value for the dataset.

        Returns:
            `None`
        """
        self.dataset_mae = dataset_mae
        self.mae_variance = self.mae - dataset_mae

    def get_accuracy_variance(self) -> float:
        """
        `get_accuracy_variance` function

        Description:
            This function returns the accuracy variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            accuracy_variance (`float`): The accuracy variance value for each instance in the dataset.
        """
        return self.accuracy_variance
    
    def get_precision_variance(self) -> float:
        """
        `get_precision_variance` function

        Description:
            This function caclulates and returns the precision variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            precision_variance (`float`): The precision variance value for each instance in the dataset.
        """
        return self.precision_variance
    
    def get_recall_variance(self) -> float:
        """
        `get_recall_variance` function

        Description:
            This function returns the recall variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            recall_variance (`float`): The recall variance value for each instance in the dataset.
        """
        return self.recall_variance
    
    def get_f1_score_variance(self) -> float:
        """
        `get_f1_score_variance` function

        Description:
            This function returns the f1_score variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            f1_score_variance (`float`): The f1_score variance value for each instance in the dataset.
        """
        return self.f1_score_variance
    
    def get_rmse_variance(self) -> float:
        """
        `get_rmse_variance` function

        Description:
            This function returns the rmse variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            rmse_variance (`float`): The rmse variance value for each instance in the dataset.
        """
        return self.rmse_variance

    def get_mse_variance(self) -> float:
        """
        `get_mse_variance` function

        Description:
            This function returns the mse variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            mse_variance (`float`): The mse variance value for each instance in the dataset.
        """
        return self.mse_variance

    def get_mae_variance(self) -> float:
        """
        `get_mae_variance` function

        Description:
            This function returns the mae variance value for each instance in the dataset.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            mae_variance (`float`): The mae variance value for each instance in the dataset.
        """
        return self.mae_variance

    def is_influential(self, model_type: ModelType) -> bool:
        """
        `is_influential` function

        Description:
            This function checks if instance is influential and returns the outcome.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.
            `model_type` (`ModelType`): The type of the model.

        Returns:
            influential (`bool`): Whether the instance is influential or not.
        """
        if model_type == ModelType.BinaryClassification or model_type == ModelType.MulticlassClassification:
            self.instance_score = abs(self.accuracy_variance) + abs(self.precision_variance) + abs(self.recall_variance) + abs(self.f1_score_variance)
            if self.instance_score > 0:
                self.influential = True
            return self.influential
        elif model_type == ModelType.Regression:
            self.instance_score = abs(self.rmse_variance) + abs(self.mse_variance) + abs(self.mae_variance)
            if self.instance_score > 0.2:
                self.influential = True
            return self.influential