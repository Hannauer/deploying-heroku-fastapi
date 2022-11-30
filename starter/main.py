import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from starter.load_and_infer import start_inference

#TO DO:
  #finishing model API

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    native_country: str

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