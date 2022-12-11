import pandas as pd
import os

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.load_and_infer import start_inference

# if "DYNO" in os.environ and os.path.isdir(".dvc"):
#     os.system("dvc config core.no_scm true")
#     if os.system("dvc pull") != 0:
#         exit("dvc pull failed")
#     os.system("rm -r .dvc .apt/usr/lib/dvc")


class InputData(BaseModel):
    age: int =  Field(..., example=21)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Not-in-family")
    race: str =Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174)
    native_country: str = Field(..., example="United-States")

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