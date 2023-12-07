import uvicorn
from fastapi import FastAPI, Path
from pydantic import BaseModel, PrivateAttr, Field, PositiveFloat, computed_field
import pandas as pd
from datetime import datetime

import commun

app = FastAPI()

def load_data(path):
    print(f"Reading train data from csv: {path}")
    trips = pd.read_csv(path)
    trips_dict = trips.to_dict(orient='records')
    return trips_dict


def predict(data):
    print("making perdictions")
    data = commun.preprocess_data(data)
    column_transformer, feature_names = commun.load_column_transformer(commun.COLUMN_TRANSFORMER_PATH)
    model = commun.load_model(commun.MODEL_PATH)

    data_transformed = pd.DataFrame(column_transformer.transform(data), columns=feature_names)

    y_pred = model.predict(data_transformed)
    y_pred = commun.postprocess_target(y_pred)
    return y_pred


class Trip(BaseModel):
    vendor_id: int
    pickup_datetime: datetime
    dropoff_datetime: datetime
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str


trips = load_data(commun.DB_PATH) # TODO: not sure about this


@app.get("/")
def root():
    return {"message" : "hello"}


@app.get("/trips/list/{count}")
def read_trips(count: int = Path(..., title="The number of trips to retrieve")):
    return {"trips": trips[:count]}

@app.post("/trips/create")
def create_trip(trip:Trip):
    print("in api app post trips/create")
    trip_data = pd.DataFrame([trip.dict()])
    print('tripdata: ', trip_data)
    predictions = predict(trip_data)
    print('predictions: ', predictions)
    return {"trip_created" : trip.dict(),
            "prediction" : predictions}


if(__name__) == '__main__':
    uvicorn.run("main:app", host = "O.O.O.O", port=5000, reload=True)

