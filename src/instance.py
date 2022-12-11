class Instance():
    """
    The `Instance` class

    Description:
        This class is used to create an instance of the `Instance` class.
    """
    
    def __init__(self):
        """ 
        Description:
            The constructor of the class `Instance`.

        Args: 
            * `self` (`Instance`): The instance of the class `Instance`.
        
        Returns:
            `None`
        """
        self.dfbeta = None
        self.rmse = None
        self.r2 = None
        self.weights = None
    
    def calculate_dfbeta(self) -> None:
        """
        `calculate_dfbeta` function

        Description:
            This function calculates the dfbeta value for each instance in the dataset.

        Args:
            `self` (`InfluentialInstance`): The instance of the class `InfluentialInstance`.

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
            `self` (`InfluentialInstance`): The instance of the class `InfluentialInstance`.

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
            `self` (`InfluentialInstance`): The instance of the class `InfluentialInstance`.

        Returns:
            `None`
        """
        pass

    def get_dfbeta(self) -> float: 
        """
        `get_dfbeta` function

        Description:
            This function returns the dfbeta value for each instance in the dataset.

        Args:
            `self` (`InfluentialInstance`): The instance of the class `InfluentialInstance`.

        Returns:
            dfbeta (`float`): The dfbeta value for each instance in the dataset.
        """
        return self.dfbeta
    
    def get_rmse(self) -> float:
        """
        `get_rmse` function

        Description:
            This function returns the rmse value for each instance in the dataset.

        Args:
            `self` (`InfluentialInstance`): The instance of the class `InfluentialInstance`.

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
            `self` (`InfluentialInstance`): The instance of the class `InfluentialInstance`.

        Returns:
            r2 (`float`): The r2 value for each instance in the dataset.
        """
        return self.r2


    