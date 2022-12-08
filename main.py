import sys
import warnings
from data import Data

print('Number of arguments: {}'.format(len(sys.argv)))
try: 
    dataset_file_name = sys.argv[1]
    data = Data(dataset_file_name=dataset_file_name)
    dataset = data.get_dataset()
    print(dataset.head())
except:
    warnings.warn('No dataset specified. Using the default dataset (breast_cancer)')
    data = Data()
    print('Dataset file name: {}'.format(data.get_dataset_file_name()))
    dataset = data.get_dataset()
    print(dataset.head())
