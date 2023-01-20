import warnings
from src.data import Data
from src.enums import ModelName, Datasets
from src.model import Model
from src.args_parser import ArgsParser
import time

args = ArgsParser().parse_args()

print("\nArguments:")
print("\tData:\t\t{0}".format(args.data))
print("\tModel:\t\t{0}".format(args.model))

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
start_time = time.time()
model.fit()
end_time = time.time()
print("Model fitted.\n")

first_results = []
# Prediction and evaluation
model.predict(model.get_data().get_X_test())
first_results.append(model.get_accuracy(forInstance=False))
first_results.append(model.get_precision(forInstance=False))
first_results.append(model.get_recall(forInstance=False))
first_results.append(model.get_f1_score(forInstance=False))
first_results.append(model.get_data().get_X_train().shape[0])
first_results.append(end_time - start_time)

found_ii = model.train_for_influential_instances()
if found_ii:
    start_time = time.time()
    model.fit_with_influential_instances()
    end_time = time.time()
    final_results = []
    # Prediction and evaluation
    model.predict(model.get_data().get_X_test())
    final_results.append(model.get_accuracy(forInstance=False))
    final_results.append(model.get_precision(forInstance=False))
    final_results.append(model.get_recall(forInstance=False))
    final_results.append(model.get_f1_score(forInstance=False))
    final_results.append(model.get_data().get_X_train().shape[0])
    final_results.append(end_time - start_time)

    print("\n\n\t\t\tInitial results: \t\t Final results:")
    print("Accuracy: \t\t{0} \t\t\t\t {1}".format(round(first_results[0], 3), round(final_results[0], 3)))
    print("Precision: \t\t{0} \t\t\t\t {1}".format(round(first_results[1], 3), round(final_results[1], 3)))
    print("Recall: \t\t{0} \t\t\t\t {1}".format(round(first_results[2], 3), round(final_results[2], 3)))
    print("F1 score: \t\t{0} \t\t\t\t {1}".format(round(first_results[3], 3), round(final_results[3], 3)))
    print("Dataset size: \t\t{0} \t\t\t\t {1}".format(first_results[4], final_results[4]))
    print("Time: \t\t\t{0} \t\t\t\t {1}".format(round(first_results[5], 3), round(final_results[5], 3)))
    print("Dataset decrease: \t{0}%".format(round((1 - final_results[4] / first_results[4]) * 100, 2)))
