
"""
This function produces the performance results for slices of the data
"""
import sys
import os
sys.path.append('./')
import pandas as pd
from starter.ml.model import  partial_inference
import pickle
from starter.constants import cat_features

def get_performance_on_partial_data():

    census_data = pd.read_csv("./data/census.csv")
    model = pickle.load(open("./model/lr_model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    lb = pickle.load(open("./model/lb.pkl", "rb"))
    for feature in cat_features:
        partial_inference(census_data,fixed_feature=feature, model=model,encoder=encoder,lb=lb)


if __name__ == '__main__':
    get_performance_on_partial_data()

