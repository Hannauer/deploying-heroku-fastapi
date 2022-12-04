import requests
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

def test_get_method():
    """
    Test the get method on fast API
    """

    client_response = client.get("/")
    assert client_response.status_code == 200
    assert client_response.son() == {"message": "Greetings!"}


def test_post_high(client):
    request = client.post("/infer/", json={"age": 31,
                                    "workclass": "Private",
                                    "fnlgt": 209642,
                                    "education": "HS-grad",
                                    "education_num": 14,
                                    "marital_status": "Married-civ-spouse",
                                    "occupation": "Exec-managerial",
                                    "relationship": "Husband",
                                    "race": "White",
                                    "sex": "Male",
                                    "capital_gain": 14084,
                                    "native_country": "United-States"})
    assert request.status_code == 200
    assert request.json() == {"prediction": ">50K"}