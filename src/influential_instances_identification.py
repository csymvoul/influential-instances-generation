import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances as md
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
        self.influential_instances = None
        self.dataset = dataset
        self.threshold_distance = None

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
                `bool`: `True` if the influential instances were identified, `False` otherwise.
        """
        print("Identifying the influential instances...")
        if len(self.influential_instances_indices) != 0:
            self.__set_influential_instances()
            self.__set_threshold_distance()
            print("Current number of influential instances: \t", len(self.influential_instances_indices))
            for i, instance in self.dataset.iterrows():
                if i not in self.influential_instances_indices:
                    # Calculate the distance to the other instances in the `influential_instances_indices
                    # If the distance is smaller than a certain threshold, then the instance is considered to be influential
                    # Add the instance's index to the `influential_instances_indices`
                    if self.threshold_distance < md(instance.values.reshape(1, -1), self.influential_instances.values).min():
                        self.influential_instances_indices.append(i)
            print("Final number of influential instances: \t\t", len(self.influential_instances_indices))
            self.__set_influential_instances()
            return True
        else:
            print("No influential instances were identified.")
            return False

    def __set_influential_instances(self) -> None:
        """
            Description: 
                This function sets the influential instances from the indices in the `influential_instances_indices`.

            Algorithm description:
                * Iterate over the `influential_instances_indices`.
                * For each index, get the instance from the `dataset`.
                * Add the instance to the `influential_instances`.

            Args:
                `None`
            
            Returns:
                `None`
        """
        self.influential_instances = pd.DataFrame()
        for index in self.influential_instances_indices:
            # Get the instance from the `dataset`
            # Add the instance to the `influential_instances`
            self.influential_instances = self.influential_instances.append(self.dataset.iloc[index])
        print("Number of influential instances: \t\t", len(self.influential_instances_indices))

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
        print("Setting the threshold distance...")
        distances = []
        for i, instance in self.influential_instances.iterrows():
            for j, instance2 in self.influential_instances.iterrows():
                if i != j:
                    # Calculate the Manhattan distance
                    # Set the threshold distance to the average of the distances
                    distances.append(md(instance.values.reshape(1, -1), instance2.values.reshape(1, -1))[0][0])
        
        self.threshold_distance = sum(distances) / len(distances)
        print("Threshold distance: \t\t\t\t", round(self.threshold_distance, 4))

    def get_influential_instances(self) -> pd.DataFrame:
        """
            Description: 
                Returns the influential instances.

            Args:
                `None`
            
            Returns:
                pd.DataFrame: The influential instances.
        """
        return self.influential_instances