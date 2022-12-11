from joblib import load
from .ml.model import compute_score_per_slice, compute_metrics

def model_validate(df_test, path):
    '''
    Validate the model using slices of the original dataframe
    
    '''

    model = load(path+'/model/model.pickle')
    encoder = load(path+'/model/one_hot_encoding.pickle')
    lb = load(path+'/model/lb.pickle')

    compute_metrics(model=model, df_test=df_test, encoder=encoder,  lb=lb, path=path)
    compute_score_per_slice(model=model, df_test=df_test, encoder=encoder,  lb=lb, path=path)