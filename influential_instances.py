from src.data import Data
from src.instance import Instance
from src.enums import Datasets
from sklearn.metrics import mean_squared_error, r2_score

class InfluentialInstance(Instance): 
    """
    `InfluentialInstance` class

    Description:
        This class implements the influential instance detection algorithm.
    Args:
        `Instance` (`Instance`): The instance of the class `Instance`.
    """

    def __init__(self) -> None:
        """ 
        Description:
            The constructor of the class `InfluentialInstance`.

        Args: 
            * `self` (`InfluentialInstance`): The instance of the class `InfluentialInstance`.
        
        Returns:
            `None`
        """
        super().__init__()
        
    
    
