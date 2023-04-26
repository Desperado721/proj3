import requests
import json

response = requests.get("https://udacity-proj3.herokuapp.com/info/")
# get
print(response.status_code)
print(response.json())

# post
test_data = [
    {
        "age": 0,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": "<=50K",
    }
]


response = requests.post(
    "https://udacity-proj3.herokuapp.com/predict/", data=json.dumps(test_data)
)
print(response.status_code)
print(response.json())
