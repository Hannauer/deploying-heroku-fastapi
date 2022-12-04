import os
import pytest
from starter.train_model import load_data, split_data, train_save_model


@pytest.fixture(scope="session")
def data():
    df =  load_data('/data/census.csv')
    return df

@pytest.fixture(scope="session")
def df_train(data):
    df_train, _ = split_data(data)

    return df_train


def test_load_data():
    '''
    Testing the loading data funciont
    '''
    data =  load_data('/data/census.csv')

    assert data.shape[0] > 0
    assert data.shape[1] > 0

def test_split_data(data):
    '''
    Testing the data spliting function
    '''
    df_train, df_test = split_data(data)

    assert df_train.shape [0] > 0
    assert df_test.shape [0] > 0


def test_train_save_model(df_train):
    '''
    Testing the train and save model
    '''

    train_save_model(df_train)

    assert os.path.isfile("../model/model.pickle")
    assert os.path.isfile("../model/one_hot_encoder.pickle")
    assert os.path.isfile("../model/lb.pickle")
