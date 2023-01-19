import pandas as pd
from src.enums import Datasets

def clean_data(dataset:pd.DataFrame, dataset_file_name:Datasets) -> None:
    """
    `clean_data` function

    Description: 
        This function performs data cleaning to the given dataset. 
        In particular, it removes `None` or empty values and duplicates.
        Also, it performs some changes to the dataset, such as replacing values or dropping columns.

    Args:
        `dataset` (`pd.DataFrame`): The dataset to clean.
        `dataset_file_name` (`Datasets`): The name of the dataset file.

    Returns:
        `None`
    """
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()

    if dataset_file_name == Datasets.BreastCancer:
        dataset.drop(['id'], axis=1, inplace=True)
        dataset['diagnosis'] = dataset['diagnosis'].replace('M', 1).replace('B', 0)
    elif dataset_file_name == Datasets.CervicalCancer:
        dataset['Dx:Cancer'] = dataset['Dx:Cancer'].replace('Yes', 1).replace('No', 0)
        dataset.drop([ 'Num_of_pregnancies',
                            'STDs:_Time_since_first_diagnosis', 
                            'STDs:_Time_since_last_diagnosis', 
                            'Number_of_sexual_partners', 
                            'First_sexual_intercourse', 
                            'Smokes', 
                            'Smokes_(years)', 
                            'Smokes_(packs/year)', 
                            'Hormonal_Contraceptives',
                            'Hormonal_Contraceptives_(years)',
                            'IUD',
                            'IUD_years',
                            'STDs',
                            'STDs_number',
                            'STDs:condylomatosis',
                            'STDs:cervical_condylomatosis',
                            'STDs:vaginal_condylomatosis',
                            'STDs:vulvo-perineal_condylomatosis',
                            'STDs:syphilis',
                            'STDs:pelvic_inflammatory_disease',
                            'STDs:genital_herpes',
                            'STDs:molluscum_contagiosum',
                            'STDs:AIDS',
                            'STDs:HIV',
                            'STDs:Hepatitis_B',
                            'STDs:HPV'], axis=1, inplace=True)
    elif dataset_file_name == Datasets.Mifem:
        dataset['outcome'] = dataset['outcome'].replace('live', 1).replace('dead', 0)
        dataset['premi'] = dataset['premi'].replace('y', 2).replace('n', 1).replace('nk', 0)
        dataset['smstat'] = dataset['smstat'].replace('c', 3).replace('x', 2).replace('n', 1).replace('nk', 0)
        dataset['diabetes'] = dataset['diabetes'].replace('y', 2).replace('n', 1).replace('nk', 0)
        dataset['highbp'] = dataset['highbp'].replace('y', 2).replace('n', 1).replace('nk', 0)
        dataset['hichol'] = dataset['hichol'].replace('y', 2).replace('n', 1).replace('nk', 0)
        dataset['angina'] = dataset['angina'].replace('y', 2).replace('n', 1).replace('nk', 0)
        dataset['stroke'] = dataset['stroke'].replace('y', 2).replace('n', 1).replace('nk', 0) 
