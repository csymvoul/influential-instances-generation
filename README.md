# Influential Instances identification

## Project description
This section describes the architecture of the proposed influence-based learning strategy. Starting with the goals, our objective is to enhance the developed ML or DL model by reducing the dataset, yet preserving the amount of information, in conformance to the infusion of causality through the data. 

Our strategy includes 5 main steps; (i) the initial training of the ML / DL algorithm with the original dataset, (ii) the calculation of $DFBETA$ and Root Mean Square Error ( $RMSE$ ) measures for each instance of the original dataset, (iii) the selection of the most influential instances according to the two measures, (iv) the identification of all influential instances using the K-Means algorithm along with the generation of the new dataset which is consisted of only the identified influential instances, and finally, (v) the re-training of the ML / DL model using the influential instances dataset.

## Dependencies

First, you need to install the dependencies / libraries using `pip`. All necessary libraries are listedin the `requirements.txt` file, found in the root. By executing the following command, all dependencies will be installed: 

    pip3 install -r requirements.txt

## Execution

The `main.py` file should be executed. There are two (optional) arguments that can be inserted: 

* `--data` where the argument should be either of the available dataset names, found in the `Datasets` enum in the `src/enums.py` file. This argument sets the dataset to be used. 
    * Default dataset if the argument is not set, or not found is the `BreastCancer` dataset.
* `--model` where the argument should be either of the available model names, found in the `ModelName` enum in the `src/enums.py` file. This argument sets the model to be used. 
    * Default model if the argument is not set, or not found is the `LogisticRegression` model.

When executing the program, an initial training is performed. Then the model is re-trained by removing every instance one at a time, trying to figure out which of those are influential to the model's results. When the initial influential instances list is found, it is enhanced with other instances similar to either one of the ones that are already influential. Once the influential instances are found, the model is re-trained.

The results of the program include information related to the initial mdoel training, as well as the training once the influential instances are found. 

## Results

Below the results of the program can be found: 

    ~: cd /influential_instances_service
    ~: python3 main.py --data BreastCancer --model RandomForestClassifier

```
Arguments:
        Data:           BreastCancer
        Model:          RandomForestClassifier

Information:
        Model type:     BinaryClassification
        Model name:     RandomForestClassifier
        Model:          <class 'sklearn.ensemble._forest.RandomForestClassifier'>
        Dataset:        BreastCancer

Fitting the model...
Model fitted.

Identifying the influential instances...
Number of influential instances:                 54
Setting the threshold distance...
Threshold distance:                              1231.9743
Current number of influential instances:         54
Final number of influential instances:           55
Number of influential instances:                 55
Training with influential instances...


                        Initial results:                 Final results:
Accuracy:               0.965                            0.974
Precision:              0.976                            0.935
Recall:                 0.93                             1.0
F1 score:               0.952                            0.966
Dataset size:           455                              55
Time:                   0.088                            0.044
Dataset size:           455                              55
Dataset decrease:       87.91%
```