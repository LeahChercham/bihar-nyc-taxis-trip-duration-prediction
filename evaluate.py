import sqlite3
import pandas as pd
from sklearn.metrics import mean_squared_error

import commun

def load_test_data(path):
    print(f"Reading test data from the database: {path}")
    data_test = pd.read_csv(path)
    X = data_test.drop(columns=['target'])
    y = data_test['target']
    return X, y

def evaluate_model(model, X, y):
    print(f"Evaluating the model")
    y_pred = model.predict(X)
    score = mean_squared_error(y, y_pred)
    return score

if __name__ == "__main__":

    X_test, y_test = load_test_data(commun.DB_PATH)
    X_test = commun.preprocess_data(X_test)
    model = commun.load_model(commun.MODEL_PATH)
    score_test = evaluate_model(model, X_test, y_test)
    print(f"Score on test data {score_test:.2f}")
