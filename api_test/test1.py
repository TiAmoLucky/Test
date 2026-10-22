import requests

headers = {"content-Type": "application/json"}
data = {"model":"deepseek-r1:7b","prompt":"你是谁?","stream":False}
response = requests.post("http://127.0.0.1:11434/api/generate",json=data,headers=headers)
print(response.json())
print(response.json().get("response",""))