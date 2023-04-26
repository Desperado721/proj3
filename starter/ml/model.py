from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
from starter.ml.data import process_data
import pickle
import numpy as np
import os
from starter.constants import cat_features



# Optional: implement hyperparameter tuning.
def model_train(X_train, y_train):
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
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr


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
    """Run model inferences and return the predictions.

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
    preds = model.predict(X)
    return preds



def partial_inference(X: pd.DataFrame, fixed_feature: str, model, encoder, lb):
    """
    This function is used to  that computes performance on model slices.
    I.e. a function that computes the performance metrics when the value of a
    given feature is held fixed. E.g. for education, it would print out the
    model metrics for each slice of data that has a particular value for education.
    You should have one set of outputs for every single unique value in education.
    """


    # Get test set: Use the same random_state as the training

    # Slice data and get performance metrics for each slice
    print("calculating sliced data metrics:{}".format(fixed_feature))
    for feature in cat_features:
        for entry in X[feature].unique():
            temp_df = X[X[feature] == entry]
            X_test, y_test, _, _ = process_data(
                temp_df, cat_features, label="salary", training=False,
                encoder=encoder, lb=lb
            )
            y_pred = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            sliced_data_path = os.path.join(os.getcwd(), "sliced_data_performance/slice_output.txt")
            with open(sliced_data_path, 'a') as file:
                file.write(f"{feature} = {entry}; Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}\n")
