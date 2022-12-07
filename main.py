import pandas as pd
import os

from fastapi import FastAPI
from pydantic import BaseModel
from starter.load_and_infer import start_inference

# if "DYNO" in os.environ and os.path.isdir(".dvc"):
#     os.system("dvc config core.no_scm true")
#     if os.system("dvc pull") != 0:
#         exit("dvc pull failed")
#     os.system("rm -r .dvc .apt/usr/lib/dvc")


class InputData(BaseModel):
    age: int = 21
    workclass: str = "State-gov"
    fnlgt: int = 77516
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Never-married"
    occupation: str = "Exec-managerial"
    relationship: str = "Not-in-family"
    race: str ="White"
    sex: str = "Male",
    capital_gain: int = 2174
    native_country: str = "United-States"

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
  return {"greeting": "Welcome!"}


@app.post("/infer/")
async def exec_infer(input_data: InputData):
  
  convert_dict = input_data.dict()
  print(convert_dict)
  data = pd.DataFrame(convert_dict, index = [1])
  print(data)

  y_pred = start_inference(data,)

  return {'preds': y_pred}