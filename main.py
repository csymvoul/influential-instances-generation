import warnings
from src.data import Data
from src.enums import ModelName, Datasets
from src.model import Model
from src.args_parser import ArgsParser

args = ArgsParser().parse_args()

print("Arguments:")
print("\tdata:\t{0}".format(args.data))
print("\tmodel:\t{0}".format(args.model))

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

print("\nInformation:")
print("\tModel type:\t{0}".format(model.get_type().value))
print("\tModel name:\t{0}".format(model.get_model_name().value))
print("\tModel:\t\t{0}".format(type(model.get_model())))
print("\tDataset:\t{0}".format(model.get_data().get_dataset_file_name().value))

model.get_data().set_instances()
data = model.get_data()

# Fit the model
print("\nFitting the model...")
model.fit()
print("Model fitted\n")


model.predict(model.get_data().get_X_test())
print("Predictions: \n{0}\n".format(model.get_predictions()))
print("Actual values: \n{0}\n".format(model.get_data().get_y_test().to_numpy()))
print("Accuracy: {0}".format(model.get_accuracy()))
