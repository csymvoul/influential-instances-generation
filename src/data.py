from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.enums import Datasets
from sklearn.model_selection import train_test_split
from src.instance import Instance
from src.influential_instance import InfluentialInstance
from src.influential_instances_identification import InfluentialInstancesIdentification

class Data:
    """
    The `Data` class.

    This class is used to load the dataset and perform data cleaning and an initial visualization.
    """

    def __init__(self, dataset_file_name: Datasets = Datasets.BreastCancer) -> None:
        """ 
        Description:
            The constructor of the `Data` class.

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
        # self.clean_data()
        self.X = None
        self.y = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.y_pred = None
        self.instances = []
        self.set_instances()
        self.train_test_split()
        self.dataset_for_influential_check = None
        self.influential_instances = []
        self.dataset_beta = None
        self.dataset_rmse = None
        self.dataset_accuracy = None
        self.dataset_precision = None
        self.dataset_recall = None
        self.dataset_f1_score = None
        self.dataset_r2 = None
        self.threshold_dfbeta = None
        self.threshold_rmse = None
        self.threshold_r2 = None
        self.threshold_accuracy = None
        self.threshold_precision = None
        self.threshold_recall = None
        self.threshold_f1_score = None
        self.dfbetas = None
        self.accuracies = None
        self.precisions = None
        self.recalls = None
        self.f1_scores = None
        self.influential_instances_identification = None
    
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
        try:
            self.path = 'datasets/'+self.dataset_file_name.value+'.csv'
            self.dataset = pd.read_csv(self.path)
        except:
            raise ValueError('Invalid dataset file name.')

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
            In particular, it removes `None` or empty values and duplicates.
            Also, it performs some changes to the dataset, such as replacing values or dropping columns.

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            `None`
        """
        self.dataset = self.dataset.dropna()
        # self.dataset = self.dataset.drop_duplicates()

        if self.dataset_file_name == Datasets.BreastCancer:
            self.dataset.drop(['id'], axis=1, inplace=True)
            self.dataset['diagnosis'] = self.dataset['diagnosis'].replace('M', 1).replace('B', 0)
        elif self.dataset_file_name == Datasets.CervicalCancer:
            self.dataset['Dx:Cancer'] = self.dataset['Dx:Cancer'].replace('Yes', 1).replace('No', 0)
            self.dataset.drop([ 'Num_of_pregnancies',
                                'STDs:_Time_since_first_diagnosis', 
                                'STDs:_Time_since_last_diagnosis', 
                                'Number_of_sexual_partners', 
                                'First_sexual_intercourse', 
                                'Smokes', 
                                'Smokes_(years)', 
                                'Smokes_(packs/year)', 
                                'Hormonal_Contraceptives',
                                'Hormonal_Contraceptives_(years)',
                                'IUD',
                                'IUD_years',
                                'STDs',
                                'STDs_number',
                                'STDs:condylomatosis',
                                'STDs:cervical_condylomatosis',
                                'STDs:vaginal_condylomatosis',
                                'STDs:vulvo-perineal_condylomatosis',
                                'STDs:syphilis',
                                'STDs:pelvic_inflammatory_disease',
                                'STDs:genital_herpes',
                                'STDs:molluscum_contagiosum',
                                'STDs:AIDS',
                                'STDs:HIV',
                                'STDs:Hepatitis_B',
                                'STDs:HPV'], axis=1, inplace=True)
        elif self.dataset_file_name == Datasets.Mifem:
            self.dataset['outcome'] = self.dataset['outcome'].replace('live', 1).replace('dead', 0)
            self.dataset['premi'] = self.dataset['premi'].replace('y', 2).replace('n', 1).replace('nk', 0)
            self.dataset['smstat'] = self.dataset['smstat'].replace('c', 3).replace('x', 2).replace('n', 1).replace('nk', 0)
            self.dataset['diabetes'] = self.dataset['diabetes'].replace('y', 2).replace('n', 1).replace('nk', 0)
            self.dataset['highbp'] = self.dataset['highbp'].replace('y', 2).replace('n', 1).replace('nk', 0)
            self.dataset['hichol'] = self.dataset['hichol'].replace('y', 2).replace('n', 1).replace('nk', 0)
            self.dataset['angina'] = self.dataset['angina'].replace('y', 2).replace('n', 1).replace('nk', 0)
            self.dataset['stroke'] = self.dataset['stroke'].replace('y', 2).replace('n', 1).replace('nk', 0) 
        elif self.dataset_file_name == Datasets.StockMarket:
            self.dataset.drop(['TradeDate'], axis=1, inplace=True)
        elif self.dataset_file_name == Datasets.Services: 
            # 'gpu' 'serverless' 'rnn' 'docker' 'fpga' 'cnn' 'edge' 'knn' 'regression' 'linearprogramming'
            self.dataset['label'] = self.dataset['label'].replace('gpu', 0).replace('serverless', 0).replace('rnn', 0).replace('docker', 0).replace('fpga', 0).replace('cnn', 0).replace('edge', 0).replace('knn', 1).replace('regression', 1).replace('linearprogramming', 1)

    def normalize_data(self) -> None:
        """
        `normalize_data` function

        Description: 
            This function normalizes the dataset.

        Args:
            `self` (`Data`): The instance of the class `Data`.

        Returns:
            `None`
        """
        self.dataset = (self.dataset - self.dataset.mean()) / self.dataset.std()

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
            self.clean_data()
            self.X = self.dataset.drop('diagnosis', axis=1)
            self.y = self.dataset['diagnosis']
        elif self.dataset_file_name == Datasets.CervicalCancer:
            self.clean_data()
            self.X = self.dataset.drop('Dx:Cancer', axis=1)
            self.y = self.dataset['Dx:Cancer']
        elif self.dataset_file_name == Datasets.Corral:
            self.X = self.dataset.drop('Class', axis=1)
            self.y = self.dataset['Class']
        elif self.dataset_file_name == Datasets.Mifem:
            self.clean_data()
            self.X = self.dataset.drop('outcome', axis=1)
            self.y = self.dataset['outcome']
        elif self.dataset_file_name == Datasets.XD6:
            self.X = self.dataset.drop('Class', axis=1)
            self.y = self.dataset['Class']
        elif self.dataset_file_name == Datasets.StockMarket:
            self.clean_data()
            self.X = self.dataset.drop('Close', axis=1)
            self.y = self.dataset['Close']
        elif self.dataset_file_name == Datasets.WineQuality:
            self.X = self.dataset.drop('quality', axis=1)
            self.y = self.dataset['quality']
        elif self.dataset_file_name == Datasets.Services: 
            self.X = self.dataset.drop('label', axis=1)
            self.y = self.dataset['label']

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
        if self.dataset_file_name == Datasets.StockMarket:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y, 
                                                                                test_size = test_size,
                                                                                shuffle=False, 
                                                                                random_state = random_state)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y, 
                                                                                test_size = test_size, 
                                                                                random_state = random_state)
    
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

    def set_X_train(self, X_train: pd.DataFrame) -> None:
        """
        `set_X_train` function

        Description:
            This function sets the training set.

        Args:
            `X_train` (`pandas.DataFrame`): The training set.

        Returns:
            `None`
        """
        self.X_train = X_train

    def set_y_train(self, y_train: pd.DataFrame) -> None:
        """
        `set_y_train` function

        Description:
            This function sets the training set.

        Args:
            `y_train` (`pandas.DataFrame`): The training set.

        Returns:
            `None`
        """
        self.y_train = y_train

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

    def set_y_pred(self, y_pred: pd.DataFrame) -> None:
        """
        `set_y_pred` function

        Description:
            This function sets the predicted labels.

        Args:
            `y_pred` (`pandas.DataFrame`): The predicted labels.

        Returns:
            `None`
        """
        self.y_pred = y_pred
    
    def get_y_pred(self) -> pd.DataFrame:
        """
        `get_y_pred` function

        Description:
            This function returns the predicted labels.

        Args:
            `None`

        Returns:
            `pandas.DataFrame`: The predicted labels.
        """
        return self.y_pred

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
            self.instances.append(Instance(instance_index=i))

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
        self.influential_instances.append(index)

    def set_dataset_mse(self, dataset_mse:float) -> None:
        """
        `set_dataset_mse` function

        Description:
            This function set the MSE of the dataset.

        Args:
            `dataset_mse` (`float`): The MSE of the dataset.

        Returns:
            `None`
        """
        self.dataset_mse = dataset_mse
    
    def get_dataset_mse(self) -> float:
        """
        `get_dataset_mse` function

        Description:
            This function returns the MSE of the dataset.

        Args:
            `None`

        Returns:
            `float`: The MSE of the dataset.
        """
        return self.dataset_mse

    def set_dataset_mae(self, dataset_mae:float) -> None:
        """
        `set_dataset_mae` function

        Description:
            This function set the MAE of the dataset.

        Args:
            `dataset_mae` (`float`): The MAE of the dataset.

        Returns:
            `None`
        """
        self.dataset_mae = dataset_mae
    
    def get_dataset_mae(self) -> float:
        """
        `get_dataset_mae` function

        Description:
            This function returns the MAE of the dataset.

        Args:
            `None`

        Returns:
            `float`: The MAE of the dataset.
        """
        return self.dataset_mae

    def set_dataset_rmse(self, dataset_rmse:float) -> None:
        """
        `set_dataset_rmse` function

        Description:
            This function set the RMSE of the dataset.

        Args:
            `dataset_rmse` (`float`): The RMSE of the dataset.

        Returns:
            `None`
        """
        self.dataset_rmse = dataset_rmse
    
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
    
    def set_dataset_r2(self, dataset_r2:float) -> None: 
        """
        `set_dataset_r2` function

        Description:
            This function sets the R2 of the dataset calculated when trained with all instances.

        Args:
            `dataset_r2` (`float`): The R2 Score of the dataset calculated when trained with all instances.

        Returns:
            `None`
        """
        self.dataset_r2 = dataset_r2

    def set_dataset_beta(self, dataset_beta:float) -> None:
        """
        `set_dataset_beta` function

        Description:
            This function sets the beta of the dataset calculated when trained with all instances.

        Args:
            `dataset_beta` (`float`): The beta of the dataset calculated when trained with all instances.

        Returns:
            `None`
        """
        self.dataset_beta = dataset_beta
    
    def get_dataset_r2(self) -> float:
        """
        `get_dataset_r2` function

        Description:
            This function returns the R2 of the dataset calculated when trained with all instances.

        Args:
            `None`

        Returns:
            `float`: The R2 of the dataset calculated when trained with all instances.
        """
        return self.dataset_r2

    def get_dataset_beta(self) -> float:
        """
        `get_dataset_beta` function

        Description:
            This function returns the beta of the dataset calculated when trained with all instances.

        Args:
            `None`

        Returns:
            `float`: The beta of the dataset calculated when trained with all instances.
        """
        return self.dataset_beta

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

    def set_dataset_accuracy(self, dataset_accuracy: float) -> None:
        """
        `set_dataset_accuracy` function

        Description:
            This function sets the accuracy of the dataset calculated when trained with all instances.

        Args:
            `dataset_accuracy` (`float`): The accuracy of the dataset calculated when trained with all instances.

        Returns:
            `None`
        """
        self.dataset_accuracy = dataset_accuracy
    
    def get_dataset_accuracy(self) -> float:
        """
        `get_dataset_accuracy` function

        Description:
            This function returns the accuracy of the dataset calculated when trained with all instances.

        Args:
            `None`

        Returns:
            `float`: The accuracy of the dataset calculated when trained with all instances.
        """
        return self.dataset_accuracy
    
    def set_dataset_precision(self, dataset_precision: float) -> None:
        """
        `set_dataset_precision` function

        Description:
            This function sets the precision of the dataset calculated when trained with all instances.

        Args:
            `dataset_precision` (`float`): The precision of the dataset calculated when trained with all instances.

        Returns:
            `None`
        """
        self.dataset_precision = dataset_precision
    
    def get_dataset_precision(self) -> float:
        """
        `get_dataset_precision` function

        Description:
            This function returns the precision of the dataset calculated when trained with all instances.

        Args:
            `None`

        Returns:
            `float`: The precision of the dataset calculated when trained with all instances.
        """
        return self.dataset_precision
    
    def set_dataset_recall(self, dataset_recall: float) -> None:
        """
        `set_dataset_recall` function

        Description:
            This function sets the recall of the dataset calculated when trained with all instances.

        Args:
            `dataset_recall` (`float`): The recall of the dataset calculated when trained with all instances.

        Returns:
            `None`
        """
        self.dataset_recall = dataset_recall
    
    def get_dataset_recall(self) -> float:
        """
        `get_dataset_recall` function

        Description:
            This function returns the recall of the dataset calculated when trained with all instances.

        Args:
            `None`

        Returns:
            `float`: The recall of the dataset calculated when trained with all instances.
        """
        return self.dataset_recall
    
    def set_dataset_f1_score(self, dataset_f1_score: float) -> None:
        """
        `set_dataset_f1_score` function

        Description:
            This function sets the F1 Score of the dataset calculated when trained with all instances.

        Args:
            `dataset_f1` (`float`): The F1 Score of the dataset calculated when trained with all instances.

        Returns:
            `None`
        """
        self.dataset_f1_score = dataset_f1_score
    
    def get_dataset_f1_score(self) -> float:
        """
        `get_dataset_f1_score` function

        Description:
            This function returns the F1 Score of the dataset calculated when trained with all instances.

        Args:
            `None`

        Returns:
            `float`: The F1 Score of the dataset calculated when trained with all instances.
        """
        return self.dataset_f1_score

    def get_influential_instances(self) -> list:
        """
        `get_influential_instances` function

        Description:
            This function returns the influential instances of the dataset.

        Args:
            `None`

        Returns:
            `list`: The influential instances of the dataset.
        """
        return self.influential_instances