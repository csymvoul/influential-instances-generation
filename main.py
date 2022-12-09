import sys
import warnings
from data import Data
from model_type import ModelType, ModelName

print('Number of arguments: {}'.format(len(sys.argv)))
try: 
    dataset_file_name = sys.argv[1]
    try: 
        model_name = ModelType(sys.argv[2])
    except:
        warnings.warn('No model type specified. Using the default model type (BinaryClassification)')
        model_type = ModelType.BinaryClassification
    data = Data(dataset_file_name=dataset_file_name)
    dataset = data.get_dataset()
    print(dataset.head())
except:
    warnings.warn('No dataset specified. Using the default dataset (breast_cancer)')
    data = Data()
    try: 
        model_type = ModelType(sys.argv[2])
    except:
        warnings.warn('No model type specified. Using the default model type (BinaryClassification)')
        model_type = ModelType.BinaryClassification
    print('Dataset file name: {}'.format(data.get_dataset_file_name()))
    dataset = data.get_dataset()
    print(dataset.head())

print(model_type)