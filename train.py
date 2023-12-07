import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


import common

def load_train_data(path):
    print(f"Reading train data from csv: {path}")
    data_train = pd.read_csv(path)
    X = data_train.drop(columns=['trip_duration'])
    y = data_train['trip_duration']
    return X, y

def fit_model(X, y):
    print(f"Fitting a model")
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    score = mean_squared_error(y, y_pred)
    print(f"Score on train data {score:.2f}")
    return model


if __name__ == "__main__":
    X_train, y_train = load_train_data(common.DB_PATH)
    X_train = common.preprocess_data(X_train)
    model = fit_model(X_train, y_train)
    common.persist_model(model, common.MODEL_PATH)