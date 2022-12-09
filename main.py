import warnings
from src.data import Data
from src.enums import ModelName, Datasets
from src.model import Model
from src.args_parser import ArgsParser

args = ArgsParser.parse_args()

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

print()
print("Model:\t\t{0}".format(model.get_model().value))
print("Model type:\t{0}".format(model.get_type().value))
print("Dataset:\t{0}".format(model.get_data().get_dataset_file_name().value))
model.get_data().set_dataset(dataset_file_name=Datasets.CervicalCancer)
print("Dataset now:\t{0}\n".format(model.get_data().get_dataset_file_name().value))