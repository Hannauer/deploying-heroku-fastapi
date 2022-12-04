"""
Heroku Api test script
"""
import requests

data =  {"age": 31,
 "workclass": "Private",
 "fnlgt": 45781,
 "education": "Masters",
 "education_num": 14,
 "marital_status": "Never-married",
 "occupation": "Prof-specialty",
 "relationship": "Not-in-family",
 "race": "White",
 "sex": "Male",
 "capital_gain": 14084,
 "native_country": "United-States"}

r = requests.post('https://censu-app.herokuapp.com/infer/', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())