from tangled_up_in_unicode import script
from fastapi import FastAPI
from pydantic import BaseModel

#TO DO:
  #clean data step
  #Train model script
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

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome fellows engeneerings!"}
