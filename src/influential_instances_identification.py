import pandas as pd

class InfluentialInstancesIdentification():
    """
    The `InfluentialInstancesIdentification` class.
    
    This class is used to identify the most influential instances in a dataset.
    """
    def __init__(self, influential_instances_list:list, dataset: pd.DataFrame) -> None: 
        """
        Description: 
            The constructor of the `InfluentialInstancesIdentification` class.

        Args:
            `influential_instances_list` (`list`): A list of influential instances. This list only contains the indices of those instances in the `dataset` that are considered to be influential. 
            `dataset` (`pd.DataFrame`): The dataset containing all instances.
        
        Returns:
            `None`
        """
        self.influential_instances_list = influential_instances_list

    def get_influential_instances_list(self) -> list:
        """
        Description: 
            Returns the list of influential instances.

        Args:
            `None`
        
        Returns:
            list: A list of influential instances.
        """
        return self.influential_instances_list

    def identify_influential_instances_list(self) -> None:
        """
        Description: 
            Identifies the influential instances out of the other instances of the `influential_instances` list. 

        Steps: 
            * Iterate over the `dataset`. 
            * For each instance that is not already in the `influential_instances`, calculate its distance to the other instances in the `influential_instances_list`.
            * If the distance is smaller than a certain threshold, then the instance is considered to be influential.


        Args:
            `None`
        
        Returns:
            `None`
        """
