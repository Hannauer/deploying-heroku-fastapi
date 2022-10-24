from joblib import load
from .ml.model import compute_score_per_slice

def model_validate(df_test):
    '''
    Validate the model using slices of the original dataframe
    
    '''

    model = load('../model/model.pickle')
    encoder = load('../model/one_hot_encoder.pickle')
    lb = load('../model/lb.pickle')

    compute_score_per_slice(model=model, df_test=df_test, encoder=encoder,  lb=lb)