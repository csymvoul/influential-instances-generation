import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Data:
    """
    Description:
        The `Data` class.

        This class is used to load the dataset and perform data cleaning and an initial visualization.
    """

    def __init__(self, dataset_file_name: str = 'breast_cancer') -> None:
        """ 
        Description:
            The constructor of the class `Data`.

        Args: 
            path (`str`): The path to the dataset. Default value is the `breast_cancer` dataset.
        
        Returns:
            `None`
        """
        self.dataset_file_name = dataset_file_name
        self.path = 'datasets/'+self.dataset_file_name+'.csv'
        self.dataset = pd.read_csv(self.path)
    
    def get_dataset_file_name(self) -> str:
        """
        `get_dataset_file_name` function
        
        Description: 
            This function returns the dataset file name.

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            dataset_file_name (`str`): The dataset file name.
        """
        return self.dataset_file_name

    def get_dataset(self) -> pd.DataFrame:
        """
        `get_dataset` function
        
        Description: 
            This function returns the dataset.

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            dataset (`pandas.DataFrame`): The dataset.
        """
        return self.dataset

    def clean_data(self) -> None:
        """
        `clean_data` function

        Description: 
            This function performs data cleaning. In particular, it removes null values and duplicates.

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            dataset (`pandas.DataFrame`): The dataset without null values and duplicates.
        """
        self.dataset = self.dataset.dropna()
        self.dataset = self.dataset.drop_duplicates()
        print(self.dataset.head())

    def visualize_data(self) -> None:
        """
        `visualize_data` function

        Description: 
            This function performs data visualization. In particular, it plots the data.

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            `None`
        """
        self.dataset.hist(bins=50, figsize=(20,15))
        plt.show()
