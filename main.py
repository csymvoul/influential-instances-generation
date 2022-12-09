import sys
import warnings
from src.data import Data
from src.enums import ModelType, ModelName, Datasets
from src.model import Model

import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument("--data", help="The dataset used for training of the model.")
parser.add_argument("--model", help="The model which will be used for training.")

args=parser.parse_args()

try: 
    dataset_file_name = Datasets(args.data)
    data = Data(dataset_file_name=dataset_file_name)
    try: 
        model_name = ModelName(args.model)
        model = Model(model=model_name, data=data)
    except:
        warnings.warn('No model specified. Using the default model (LogisticRegression)')
        model = Model(data=data)
except:
    warnings.warn('No dataset specified. Using the default dataset (breast_cancer)')
    try: 
        model_name = ModelName(args.model)
        model = Model(model=model_name)
    except:
        warnings.warn('No model type specified. Using the default model (LogisticRegression)', category=DeprecationWarning)
        model = Model()
    
print(model.get_model().value)
print(model.get_type().value)
print(model.get_data().get_dataset_file_name())
model.get_data().set_dataset(dataset_file_name=Datasets.CervicalCancer)
print("Now the dataset is: ", model.get_data().get_dataset_file_name())