import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


import commun

def load_train_data(path):
    print(f"Reading train data from csv: {path}")
    data_train = pd.read_csv(path)
    X = data_train.drop(columns=['trip_duration'])
    y = data_train['trip_duration']
    return X, y

def fit_model(X, y):
    print(f"Fitting a model")
    
    # remove outliers for the training data
    X, y = commun.step4_remove_outliers(X, y)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    score = mean_squared_error(y, y_pred)
    print(f"Score on train data {score:.2f}")
    return model

def fit_column_transformer(X):
    print(f"Fitting the column transformer")
    num_features = ['log_distance_haversine', 'hour', 'abnormal_period', 'is_high_traffic_trip', 'is_high_speed_trip',
                    'is_rare_pickup_point', 'is_rare_dropoff_point']
    cat_features = ['weekday', 'month']

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)
    ])

    column_transformer.fit(X[num_features + cat_features])
    return column_transformer


if __name__ == "__main__":
    X_train, y_train = load_train_data(commun.DB_PATH)
    X_train = commun.preprocess_data(X_train)
    
    # fit and persist the column transformer
    ct = fit_column_transformer(X_train)
    commun.perist_column_transformer(ct, commun.COLUMN_TRANSFORMER_PATH)
    
    #Apply the fitted column transformer to X_train
    X_train_transformed = ct.transform(X_train)
    
    model = fit_model(X_train_transformed, y_train)
    commun.persist_model(model, commun.MODEL_PATH)