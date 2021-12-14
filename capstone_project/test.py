import requests

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://vxj60v25jl.execute-api.us-west-2.amazonaws.com/test/predict'

data = {'url': 'test_black.jpg'}

result = requests.post(url, json=data).json()
print(result)


