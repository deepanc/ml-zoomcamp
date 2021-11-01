import requests

url = 'http://localhost:9696/predict'

user_id = '2478961'
user = {
    "impression": "1",
    "ad_id": "21997940",
    "advertiser_id": "37937",
    "depth": "3",
    "position": "3",
    "query_id": "4820",
    "keyword_id": "745",
    "title_id":"2036",
    "description_id": "2252"
}

response = requests.post(url, json=user).json()
print(response)