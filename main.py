import sys
import warnings
from src.data import Data
from src.models_enum import ModelType, ModelName
from src.model import Model

print('Number of arguments: {}'.format(len(sys.argv)))

try: 
    dataset_file_name = sys.argv[1]
    data = Data(dataset_file_name=dataset_file_name)
    try: 
        model_name = ModelName(sys.argv[2])
        model = Model(model=model_name, data=data)
    except:
        warnings.warn('No model specified. Using the default model (LogisticRegression)')
        model = Model(data=data)
except:
    warnings.warn('No dataset specified. Using the default dataset (breast_cancer)')
    try: 
        model_name = ModelName(sys.argv[2])
        model = Model(model=model_name)
    except:
        warnings.warn('No model type specified. Using the default model (LogisticRegression)')
        model = Model()
    
print(model.get_model().value)
print(model.get_type().value)
print(model.get_data().visualize_data())