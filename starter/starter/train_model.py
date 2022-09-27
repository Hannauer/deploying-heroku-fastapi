# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from .ml.data import process_data
from .ml.model import train_model
from joblib import dump

import pandas as pd

# Add the necessary imports for the starter code.

# Add code to load in the data.
def load_data(csv_path):
    '''
    Read the csv file from csv_path

    Inputs
    ------
    csv_path : str
               System path to the csv file to be read
    
    Returns
    -------
    data : pd.DataFrame
           Dataframe with the data
    '''
    data = pd.read_csv(csv_path)
    return data

# Optional enhancement, use K-fold cross validation instead of a train-test split.
def split_data(data):
    '''
    Split data for train and teste

    Inputs
    ------
    data : pd.DataFrame
           Dataframe with data for training and test de model

    Outputs
    -------
    df_train : pd.DataFrame
               Dataframe with the train data
    df_test : pd.Dataframe
              Dataframe with the teste data
    '''
    df_train, df_test = train_test_split(data, test_size=0.20)

    return df_train, df_test


def train_save_model(df_train, df_test):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        df_train, categorical_features=cat_features, label="salary", training=True
    )
    # Train and save a model.
    model = train_model(X_train, y_train )

    dump(model, '../model/model.pickle')
    dump(encoder, '../model/one_hot_encoding.pickle')
    dump(lb, '../model/lb.pickle')