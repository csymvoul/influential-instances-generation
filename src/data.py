from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.enums import Datasets
from sklearn.model_selection import train_test_split
from src.instance import Instance
from src.influential_instance import InfluentialInstance

class Data:
    """
    The `Data` class.

    This class is used to load the dataset and perform data cleaning and an initial visualization.
    """

    def __init__(self, dataset_file_name: Datasets = Datasets.BreastCancer) -> None:
        """ 
        Description:
            The constructor of the class `Data`.

        Args: 
            * dataset_file_name (`Datasets`): 
                * The file name of the chosed dataset. 
                * Default value is the `Datasets.BreastCancer` dataset.
        
        Returns:
            `None`
        """
        self.dataset_file_name = dataset_file_name
        self.path = 'datasets/'+self.dataset_file_name.value+'.csv'
        self.dataset = pd.read_csv(self.path)
        self.X = None
        self.y = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.instances = None
        self.set_instances()
        self.influential_instances = []
        self.dataset_beta = None
        self.dataset_rmse = None
        self.dataset_r2 = None
        self.threshold_dfbeta = None
        self.threshold_rmse = None
        self.threshold_r2 = None
        self.dfbetas = None
    
    def get_dataset_file_name(self) -> Datasets:
        """
        `get_dataset_file_name` function
        
        Description: 
            This function returns the dataset file name.

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            dataset_file_name (`Datasets`): The dataset file name.
        """
        return self.dataset_file_name

    def set_dataset(self, dataset_file_name: Datasets) -> None:
        """
        `set_dataset` function

        Description:
            This function sets the dataset to be used for training.

        Args:
            dataset_file_name (`Datasets`): The file name of the chosed dataset.

        Raises:
            `ValueError`: If the dataset file name is not valid.
        
        Returns:
            `None`
        """
        self.dataset_file_name = dataset_file_name
        self.path = 'datasets/'+self.dataset_file_name.value+'.csv'
        self.dataset = pd.read_csv(self.path)

    def get_dataset(self) -> pd.DataFrame:
        """
        `get_dataset` function
        
        Description: 
            This function returns the dataset used for training.

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
            This function performs data cleaning to the given dataset. 
            In particular, it removes null values and duplicates.

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            dataset (`pandas.DataFrame`): The dataset without null values and duplicates.
        """
        self.dataset = self.dataset.dropna()
        self.dataset = self.dataset.drop_duplicates()
        print(self.dataset.head())
    
    def split_dataset(self) -> None:
        """
        `split_dataset` function

        Description: 
            This function splits the dataset into features and labels depending on the dataset. 

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            `None`
        """
        if self.dataset_file_name == Datasets.BreastCancer:
            self.dataset['diagnosis'] = self.dataset['diagnosis'].map({'M': 1, 'B': 0})
            self.X = self.dataset.drop('diagnosis', axis=1)
            self.y = self.dataset['diagnosis']
        elif self.dataset_file_name == Datasets.CervicalCancer:
            self.dataset['Dx:Cancer'] = self.dataset['Dx:Cancer'].map({'Yes': 1, 'No': 0})
            self.X = self.dataset.drop('Dx:Cancer', axis=1)
            self.y = self.dataset['Dx:Cancer']

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        `train_test_split` function

        Description: 
            This function splits the dataset into training and testing sets.

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            `None`
        """
        self.split_dataset()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y, 
                                                                                test_size = test_size, 
                                                                                random_state = random_state)
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
    
    def get_X(self) -> pd.DataFrame:
        """
        `get_X` function

        Description:
            This function returns the features.

        Args:
            `None`

        Returns:
            `pandas.DataFrame`: The features.
        """
        return self.X
    
    def get_y(self) -> pd.DataFrame:
        """
        `get_y` function

        Description:
            This function returns the labels.

        Args:
            `None`

        Returns:
            `pandas.DataFrame`: The labels.
        """
        return self.y

    def get_X_train(self) -> pd.DataFrame:
        """
        `get_X_train` function

        Description:
            This function returns the training set.

        Args:
            `None`

        Returns:
            `pandas.DataFrame`: The training set.
        """
        return self.X_train
    
    def get_X_test(self) -> pd.DataFrame:
        """
        `get_X_test` function

        Description:
            This function returns the testing set.

        Args:
            `None`

        Returns:
            `pandas.DataFrame`: The testing set.
        """
        return self.X_test
    
    def get_y_train(self) -> pd.DataFrame:
        """
        `get_y_train` function

        Description:
            This function returns the training set.

        Args:
            `None`

        Returns:
            `pandas.DataFrame`: The training set.
        """
        return self.y_train
    
    def get_y_test(self) -> pd.DataFrame:
        """
        `get_y_test` function

        Description:
            This function returns the testing set.

        Args:
            `None`

        Returns:
            `pandas.DataFrame`: The testing set.
        """
        return self.y_test

    def set_instances(self) -> None:
        """
        `set_instances` function

        Description:
            This function sets an Instance object for each instance of the dataset to a list.

        Args:
            `None`

        Returns:
            `None`
        """
        self.instances = []
        for i in range(len(self.dataset)):
            self.instances.append(Instance(instance_data=self.dataset.iloc[i]))
        count = 0

    def get_instances(self) -> list[Instance]:
        """
        `get_instances` function

        Description:
            This function returns the list of instances. Each instance is an Instance object.

        Args:
            `None`

        Returns:
            `list[Instance]`: The list of instances.
        """
        return self.instances
    
    def get_instance(self, index: int) -> Instance:
        """
        `get_instance` function

        Description:
            This function returns an instance. The instance is an Instance object.

        Args:
            `index` (`int`): The index of the instance.

        Returns:
            `Instance`: The instance.
        """
        return self.instances[index]

    def set_instance_as_influential(self, index: int) -> None:
        """
        `set_instance_as_influential` function

        Description:
            This function sets an instance as influential.

        Args:
            `index` (`int`): The index of the instance.

        Returns:
            `None`
        """
        influential_instance = InfluentialInstance(instance=self.instances[index])
        self.influential_instances.append(influential_instance)

    def calculate_dataset_rmse(self) -> None:
        """
        `calculate_dataset_rmse` function

        Description:
            This function calculates the RMSE of the dataset.

        Args:
            `None`

        Returns:
            `None`
        """
        self.dataset_rmse = sqrt(mean_squared_error(self.y, self.dataset['prediction']))
    
    def get_dataset_rmse(self) -> float:
        """
        `get_dataset_rmse` function

        Description:
            This function returns the RMSE of the dataset.

        Args:
            `None`

        Returns:
            `float`: The RMSE of the dataset.
        """
        return self.dataset_rmse
    
    def calculate_dataset_r2(self) -> None: 
        """
        `calculate_dataset_r2` function

        Description:
            This function calculates the R2 of the dataset.

        Args:
            `None`

        Returns:
            `None`
        """
        self.dataset_r2 = r2_score(self.y, self.dataset['prediction'])

    def calculate_dataset_beta(self) -> None:
        """
        `calculate_dataset_beta` function

        Description:
            This function calculates the beta of the dataset.

        Args:
            `None`

        Returns:
            `None`
        """
        self.dataset_beta = self.dataset['prediction'].corr(self.y)
    
    def calculate_dfbetas(self) -> None:
        """
        `calculate_dfbetas` function

        Description:
            This function calculates the dfbetas for each instance of the dataset.

        Args:
            `None`

        Returns:
            `None`
        """
        for instance in self.instances:
            instance.calculate_dfbetas()
            self.dfbetas.append(instance.get_dfbeta())

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
