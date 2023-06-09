# Script to train machine learning model.
import sys
sys.path.append("..")
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import model_train, inference, compute_model_metrics
from starter.constants import cat_features


# Add the necessary imports for the starter code.
# if "DYNO" in os.environ and os.path.isdir(".dvc"):
#     os.system("dvc config core.no_scm true")
#     if os.system("dvc pull") != 0:
#         exit("dvc pull failed")
#     os.system("rm -r .dvc .apt/usr/lib/dvc")
# Add code to load in the data.
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
data_path = os.path.join(PARENT_DIR, "data")

census_data = pd.read_csv(os.path.join(data_path, "census.csv"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(census_data, test_size=0.20)


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    encoder=encoder,
    lb=lb,
    training=False,
)


# Train and save a model.
lr = model_train(X_train, y_train)
y_preds = inference(lr, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
print(
    "precision: {:0.3f}, recall: {:0.3f}, fbeta: {:0.3f}".format(
        precision, recall, fbeta
    )
)
data_path = os.path.join(PARENT_DIR, "model")
filename = "lr_model.pkl"
pickle.dump(lr, open(os.path.join(data_path, filename), "wb"))
