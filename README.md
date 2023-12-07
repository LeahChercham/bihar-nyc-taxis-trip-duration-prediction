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

# Done: 

- Added original Notebook
- Created requirements file based on other project
- Created venv following this tutorial: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env
  * To activate venv type in terminal: venv/Scripts/activate
  * To deactivate venv type in terminal: deactivate
- Installed dependencies to venv using in terminal : pip install -r requirements.txt
- Created config files in ini and yml
- Created FastAPI main.py file basic


## Next To Do's

### debugging logic
- verify why train shows such a bad score

### model serving with fast api
- add to fastapi post request with data and answer with prediction
- use validation
- persist predictions

