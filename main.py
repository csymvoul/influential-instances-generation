import warnings
from src.data import Data
from src.enums import ModelName, Datasets
from src.model import Model
from src.args_parser import ArgsParser

args = ArgsParser().parse_args()

print("Arguments:")
print("data:\t{0}".format(args.data))
print("model:\t{0}".format(args.model))

try:
    # Dataset specified 
    dataset_file_name = Datasets(args.data)
    data = Data(dataset_file_name=dataset_file_name)
    try: 
        # Model specified
        model_name = ModelName(args.model)
        model = Model(model_name=model_name, data=data)
    except:
        # Model not specified
        warnings.warn('Model not specified. Using the default model (LogisticRegression)')
        model = Model(data=data)
except:
    pass
    # Dataset not specified
    warnings.warn('Dataset not specified. Using the default dataset (breast_cancer)')
    try: 
        # Model specified
        model_name = ModelName(args.model)
        model = Model(model_name=model_name)
    except:
        # Model not specified
        warnings.warn('Model type not specified. Using the default model (LogisticRegression)', category=DeprecationWarning)
        model = Model()

print()
print("Model type:\t{0}".format(model.get_type().value))
print("Model name:\t{0}".format(model.get_model_name().value))
print("Model:\t\t{0}".format(type(model.get_model())))

print("Dataset:\t{0}".format(model.get_data().get_dataset_file_name().value))

model.get_data().set_instances()
data = model.get_data()

# Fit the model
print("\nFitting the model...")
model.fit()
print("Model fitted\n")
print("Model coefficients: \n{0}\n".format(model.get_weights()))
model.predict(model.get_data().get_X_test())
model.calculate_rmse()
model.calculate_beta()
print("RMSE: {0}".format(model.get_rmse()))
print("Dataset BETA: {0}".format(model.get_data().get_dataset_beta()))