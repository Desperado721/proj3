# Script to train machine learning model.
import sys
import pickle
import pandas as pd
sys.path.append("/Users/jielyu/udacity/mle/proj3/")

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import model_train, inference, compute_model_metrics


# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv(
    "./data/census.csv"
)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
filename = "lr_model.pkl"
pickle.dump(lr, open('model/'+filename, "wb"))
