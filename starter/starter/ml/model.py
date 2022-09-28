from sklearn.metrics import fbeta_score, precision_score, recall_score
from .data import process_data
from sklearn.ensemble import RandomForestClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_preds = model.predict(X)
    return y_preds

def compute_score_per_slice(model, df_test, encoder,
                            lb):
    """
    Compute score per category class slice
    Parameters
    ----------
    model : sklearn.RandomForestClassifier
            Trained RandomForestClassifier
    df_test : pd.DataFrame
              Datrame with the test data
    encoder : sklearn.OneHotEncoder
              Trained encoder
    lb : sklearn.LabelEncoder
         Sklearn trained label encoder
    Returns
    -------
    """
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


    with open('../../model/slice_output.txt', 'w') as file:
        for category in cat_features:
            for cls in df_test[category].unique():
                temp_df = df_test[df_test[category] == cls]

                x_test, y_test, _, _ = process_data(
                    temp_df,
                    categorical_features=cat_features,
                    training=False,
                    label="salary", encoder=encoder, lb=lb)

                y_pred = model.predict(x_test)

                prc, rcl, fb = compute_model_metrics(y_test, y_pred)

                metric_info = "[%s]-[%s] Precision: %s " \
                              "Recall: %s FBeta: %s" % (category, cls,
                                                        prc, rcl, fb)
                file.write(metric_info + '\n')