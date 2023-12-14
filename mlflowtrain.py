# import necessary libraries
import os.path

import mlflow
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
import commun
import train
import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set tracking server uri for logging
# mlflow.set_tracking_uri(commun.TRACKING_URI)

# Create a new MLflow Experiment
# mlflow.set_experiment("MLflow Taxi Trip Duration")

# Loading Data
def loading_data():
    X_train, y_train = commun.load_data(commun.DB_PATH_TRAIN)
    X_test, y_test = commun.load_data(commun.DB_PATH_TEST)
    return X_train, y_train, X_test, y_test



def log_model(model, preprocessing_column_transformer, features_names, X_test, y_test):
    # Je fais le training de mon cote, je n'enregistre plus le model sur mon pc.
    # je garde le model et le column_transformateur et je les enregistre sur mlflow
    # cette fonction sera seulement le log
    # levaluation se fait automatiquement
    # renommer artifact pour column transformer

    X_test = commun.preprocess_data(X_test)
    y_test = commun.preprocess_target(y_test)

    # y_pred = model.predict(X_test)

    # Infer an MLflow model signature from the training data (input),
    # model predictions (output) and parameters (for inference).
    signature = infer_signature(X_train, y_train)

    # Log artifact
    # save artifact in a temp file, then load the preprocessing_column_transformer here, put it into mlflow, and delete it again from the temp file
    artifact_info = mlflow.log_artifact(
        sk_model=preprocessing_column_transformer,
        artifact_path="sklearn-artifact",
        name="column-transformer"
    )

    # Log model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature)


    # Log metrics & artifacts to MLflow tracking server
    results = mlflow.evaluate(
        model_info.model_uri,  # model_info = pointeur vers le modele
        data=pd.concat([X_test, y_test], axis=1),  # pour garder le dataframe
        targets=commun.TARGET,
        model_type="regressor",
        evaluators=["default"]
    )
    return results


def calc_scores_linearmodel(actual, pred):
    rmse = mean_squared_error(actual, pred, squared=False)
    r2 = r2_score(actual, pred)
    return {
        "rmse": rmse,
        "r2": r2}



if __name__ == "__main__":
    # pour changer le dossier dans le quel je stoque les MLRuns. Si je ne fais pas, ça créer MLRuns par défaut
    mlflow.set_tracking_uri("file:" + os.path.abspath(commun.DIR_MLRUNS))

    np.random.seed(commun.RANDOM_STATE)

    # Loading Data and Preprocessing (preprocessing for test data in log function)
    X_train, y_train, X_test, y_test = loading_data()
    X_train = commun.preprocess_data(X_train)
    y_train = commun.preprocess_target(y_train)
    # remove outliers for the training data
    X_train, y_train = commun.step4_remove_outliers(X_train, y_train)

    # création d'expériment
    exp_name = "trip_duration_prediction2"  # créer expériment avec un nom qui nous plait
    experiment_id = mlflow.create_experiment(exp_name)  # récupérer l'id de l'expériment juste créer
    # info: à l'intérieur d'un expériment, il y a des runs, et dans les runs il peut y avoir des nested runs
    mlflow.set_experiment(exp_name)

    # création d'un run
    run_name = "linear_regression"
    run_id = None  # pour l'instant pas de ID encore car le run n'a pas encore été effectué


    # TODO: artifact = column transformer
    preprocessing_column_transformer, features_names = commun.load_column_transformer(commun.COLUMN_TRANSFORMER_PATH)

    # TODO: model = linear regression
    model = commun.load_model(commun.MODEL_PATH)

    # with start run = on créer la connection à mlflow et à la fin on referme la connection
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id, description=run_name) as run:  # as run could have been whatever as runningman , as huhuhuhu ...
        results = log_model(model, preprocessing_column_transformer, features_names, X_train, X_test)
        run_id = run.info.run_id
    # fin du block ici le run nommé run est détruit


    # accés au modele
    model_uri = f"runs:/{run_id}/sklearn-model"
    # register_model = enregistrer modele sur mlflow
    mv = mlflow.register_model(model_uri, "trip_duration_prediction")
    print("Model saved to registry:")
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    print(f"Source: {mv.source}")

    # affichage d'un autre URI
    # je récupère le modele avec load_model
    print(f"models:/{commun.MODEL_NAME}/{commun.MODEL_VERSION}")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{commun.MODEL_NAME}/{commun.MODEL_VERSION}")
    y_pred = model.predict(X_test)
    print(calc_scores_linearmodel(y_test, y_pred))

