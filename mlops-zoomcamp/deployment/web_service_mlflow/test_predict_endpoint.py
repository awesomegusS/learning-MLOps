# test predict endpoint works
import requests

ride = {
    'PULocationID': 10,
    'DOLocationID': 50, 
    'trip_distance': 40, 
}
url = 'http://localhost:9696/predict'

# send request and get repsonse from model endpoint

response = requests.post(url, json=ride)
print(response.json())

