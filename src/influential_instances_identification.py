import pandas as pd


class InfluentialInstancesIdentification():
    """
        The `InfluentialInstancesIdentification` class.
        
        This class is used to identify the most influential instances in a dataset.
    """
    def __init__(self, influential_instances_indices:list, dataset: pd.DataFrame) -> None: 
        """
            Description: 
                The constructor of the `InfluentialInstancesIdentification` class.

            Args:
                `influential_instances_indices` (`list`): A list of influential instances' indices. This list only contains the indices of those instances in the `dataset` that are considered to be influential. 
                `dataset` (`pd.DataFrame`): The dataset containing all instances.
            
            Returns:
                `None`
        """
        self.influential_instances_indices = influential_instances_indices
        self.influential_instances = pd.DataFrame()
        self.dataset = dataset
        self.threshold_distance = None
        self.__set_influential_instances()

    def get_influential_instances_indices(self) -> list:
        """
            Description: 
                Returns the list of influential instances' indices.

            Args:
                `None`
            
            Returns:
                list: A list of the indices of the most influential instances.
        """
        return self.influential_instances_indices

    def identify_influential_instances(self) -> None:
        """
            Description: 
                This function identifies the influential instances out of the other instances of the `influential_instances` list. 

            Algorithm description:
                * Iterate over the `dataset`. 
                * For each instance that is not already in the `influential_instances`, calculate its distance to the other instances in the `influential_instances_indices`.
                * If the distance is smaller than a certain threshold, then the instance is considered to be influential.
                * Add the instance's index to the `influential_instances_list`.

            Args:
                `None`
            
            Returns:
                `None`
        """
        for i, instance in self.dataset.iterrows():
            if i not in self.influential_instances_indices:
                # Calculate the distance to the other instances in the `influential_instances_indices
                # If the distance is smaller than a certain threshold, then the instance is considered to be influential
                # Add the instance's index to the `influential_instances_indices`
                pass

    def __set_influential_instances(self) -> None:
        """
            Description: 
                This function sets the influential instances.

            Algorithm description:
                * Iterate over the `influential_instances_indices`.
                * For each index, get the instance from the `dataset`.
                * Add the instance to the `influential_instances`.

            Args:
                `None`
            
            Returns:
                `None`
        """
        for index in self.influential_instances_indices:
            self.influential_instances = self.influential_instances.append(self.dataset.iloc[index])

    def __set_threshold_distance(self) -> None:
        """
            Description: 
                This function sets the threshold distance. 
            
            Algorithm description:
                * Calculate the Manhattan distance among the instances of the `influential_instances`.
                * Set the threshold distance to the average of the distances.

            Args:
                `None`
            
            Returns:
                `None`
        """
        


