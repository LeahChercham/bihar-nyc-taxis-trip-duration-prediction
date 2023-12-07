# import necessary libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import commun


def load_train_data(path):
    # Import data
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
    y_pred = commun.postprocess_target(y_pred) # because we log in preprocessing
    y = commun.postprocess_target(y)
    score = mean_squared_error(y, y_pred, squared=False) # squared false for RMSE - in seconds
    whole_minutes = score // 60
    remaining_seconds = score % 60
    print(f"Score on data in seconds {score:.2f} OR in minutes: {whole_minutes} min and {remaining_seconds} seconds ")
    return model


def fit_column_transformer(X, num_features, cat_features):
    print(f"Fitting the column transformer")


    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)
    ])

    column_transformer.fit(X[num_features + cat_features])
    feature_names = column_transformer.get_feature_names_out()

    return column_transformer, feature_names


if __name__ == "__main__":
    X_train, y_train = load_train_data(commun.DB_PATH_TRAIN)
    X_train = commun.preprocess_data(X_train)
    y_train = commun.preprocess_target(y_train) # log

    # remove outliers for the training data
    X_train, y_train = commun.step4_remove_outliers(X_train, y_train )

    # features used to train the model
    num_features = ['log_distance_haversine',
                    'hour',
                    'abnormal_period',
                    'is_high_traffic_trip',
                    'is_high_speed_trip',
                    'is_rare_pickup_point',
                    'is_rare_dropoff_point']
    cat_features = ['weekday', 'month']


    # fit and persist the column transformer
    ct, feature_names = fit_column_transformer(X_train, num_features, cat_features)
    commun.persist_column_transformer(ct, feature_names, commun.COLUMN_TRANSFORMER_PATH)
    
    # Apply the fitted column transformer to X_train
    X_train_transformed = pd.DataFrame(ct.transform(X_train), columns=feature_names)

    ## at step 7.1
    model = fit_model(X_train_transformed[feature_names], y_train)
    commun.persist_model(model, commun.MODEL_PATH)
