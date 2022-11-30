from joblib import load
from .ml.data import process_data
from .ml.model import inference


def start_inference(data):
    '''
    Load model and other artifacts and run the infarece on new data

    Params
    ------
    data: : np.array
        Data used for prediction.

    Returns
    -------
    Predictions for the given data
    '''

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    model = load('model/model.pickle')
    encoder = load('model/one_hot_encoding.pickle')
    lb = load('model/lb.pickle')

    X, _, _, _ = process_data(X = data,
                            categorical_features=cat_features,
                            encoder=encoder,
                            lb=lb,
                            training=False)

    y_preds = inference(model, X)
    y_preds = lb.inverse_transform(y_preds)[0]

    return y_preds