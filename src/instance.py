from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

class Instance():
    """
    The `Instance` class

    Description:
        This class is used to create an instance of the `Instance` class.
    """

    def __init__(self, instance_data: pd.Series) -> None:
        """ 
        Description:
            The constructor of the class `Instance`.

        Args: 
            * `self` (`Instance`): The instance of the class `Instance`.
        
        Returns:
            `None`
        """
        self.instance_data = instance_data
        self.beta = None
        self.rmse = None
        self.r2 = None
        self.weights = None
        self.overall_beta = None
        self.dfbeta = None
    
    def get_instance_data(self) -> pd.Series:
        """
        `get_instance_data` function

        Description:
            This function returns the instance data.

        Args:
            `self` (`Instance`): The instance of the class `Instance`.

        Returns:
            instance_data (`pd.Series`): The instance data.
        """
        return self.instance_data

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

    