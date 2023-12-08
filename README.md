# bihar-nyc-taxis-trip-duration-prediction

This repo contains the prediction model for the trip duration in NYC created in the context of the Big Data and AI Studies at ESTIA in Bidart (France)


# How to run
## Run model
in terminal: 
* venv/Scripts/activate
* python data_separation_script.py
* python train.py
* python evaluate.py
* deactivate

## Run FastAPI
In terminal (also in venv)
* uvicorn main:app --reload 
* Test http://127.0.0.1:8000/docs# POST with this data:
<br>
{
  "vendor_id": 1,
  "pickup_datetime": "2023-12-07T12:48:41.899Z",
  "dropoff_datetime": "2023-12-07T12:48:41.900Z",
  "passenger_count": 2,
  "pickup_longitude": -73.9804,
  "pickup_latitude": 40.7644,
  "dropoff_longitude": -74.0059,
  "dropoff_latitude": 40.7128,
  "store_and_fwd_flag": "N"
}
<br>

* Results in:
<br>
[
  {
    "0": 241959266.74313968,
    "vendor_id": 1,
    "pickup_datetime": "2023-12-07T12:48:41.899000+00:00",
    "dropoff_datetime": "2023-12-07T12:48:41.900000+00:00",
    "passenger_count": 2,
    "pickup_longitude": -73.9804,
    "pickup_latitude": 40.7644,
    "dropoff_longitude": -74.0059,
    "dropoff_latitude": 40.7128,
    "store_and_fwd_flag": "N"
  }
]
<br>

# Done: 

- Added original Notebook
- Created requirements file based on other project
- Created venv following this tutorial: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env
  * To activate venv type in terminal: venv/Scripts/activate
  * To deactivate venv type in terminal: deactivate
- Installed dependencies to venv using in terminal : pip install -r requirements.txt
- Created config files in ini and yml
- Created FastAPI main.py file basic
- Created FastAPI post request handler and answer with prediction and trip


## Next To Do's

### debugging 
- FastAPI: why is the postprocessing (expm1) resulting in overflow errors. Maybe executed at the wrong place

### model serving with fast api
- use validation
- persist predictions

### MLFLOW

