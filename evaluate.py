import sqlite3
import pandas as pd
from sklearn.metrics import mean_squared_error

import commun


def load_test_data(path):
    print(f"Reading test data from the database: {path}")
    data_test = pd.read_csv(path)
    X = data_test.drop(columns=['trip_duration'])
    y = data_test['trip_duration']
    return X, y


def evaluate_model(model, X, y):
    print(f"Evaluating the model")
    y_pred = model.predict(X)
    y_pred = commun.postprocess_target(y_pred)
    y = commun.postprocess_target(y)
    score = mean_squared_error(y, y_pred, squared=False)
    whole_minutes = score // 60
    remaining_seconds = score % 60
    print(
        f"Score on test data in seconds {score:.2f} OR in minutes: {whole_minutes} min and {remaining_seconds} seconds ")

    return score


if __name__ == "__main__":
    X_test, y_test = load_test_data(commun.DB_PATH_TEST)
    X_test = commun.preprocess_data(X_test)
    # add model column transformer
    column_transformer, feature_names = commun.load_column_transformer(commun.COLUMN_TRANSFORMER_PATH)
    model = commun.load_model(commun.MODEL_PATH)

    # Apply the column transformer to X_test and convert to DataFrame
    X_test_transformed = pd.DataFrame(column_transformer.transform(X_test), columns=feature_names)

    y_test = commun.preprocess_target(y_test)
    score_test = evaluate_model(model, X_test_transformed, y_test)
    whole_minutes = score_test // 60
    remaining_seconds = score_test % 60
    print(f"Score on test data in seconds: {score_test:.2f} OR in minutes: {whole_minutes} min and {remaining_seconds} seconds")
