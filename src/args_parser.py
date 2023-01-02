import argparse, textwrap

class ArgsParser:
    """
    `ArgsParser` class.

    This class is used for parsing the arguments passed to the program.
    """
    @staticmethod
    def parse_args():
        """
        parse_args function

        Description:
            This function parses the arguments passed to the program.

        Args:
            `None`

        Returns:
            `argparse.Namespace`: The parsed arguments.
        """
        parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument("--data", 
                            help=textwrap.dedent('''\
                                The dataset used for training of the model. 
                                The available datasets are the following:
                                    * BreastCancer 
                                    * CervicalCancer 
                                    * XD6 
                                    * Mifem
                                    * Corral'''))
        parser.add_argument("--model", 
                            help=textwrap.dedent('''\
                                The model which will be used for training.
                                The available models are the following: 
                                    * LogisticRegression 
                                    * KNeighborsClassifier 
                                    * SVC 
                                    * KMeans 
                                    * DecisionTreeClassifier 
                                    * RandomForestClassifier 
                                    * GradientBoostingClassifier 
                                    * XGBClassifier 
                                    * CatBoostClassifier 
                                    * MLPClassifier 
                                    * GaussianNB 
                                    * LinearDiscriminantAnalysis 
                                    * QuadraticDiscriminantAnalysis 
                                    * LinearRegression'''))

        return parser.parse_args()