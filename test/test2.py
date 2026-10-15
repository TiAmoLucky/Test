import requests

url = "http://127.0.0.1/v1/chat-messages"

headers = {
    "Authorization": "Bearer app-QJUiV1w54y7GLD6n8UFBABBO",   # ← 换成你的 Key
    "Content-Type": "application/json"
}

data = {
    # "conversation_id": "...",   # 第一次不要这行
    "inputs": {},
    "query": "你好",
    "response_mode": "blocking",
    "user": "abc-123"
}

r = requests.post(url, json=data, headers=headers)

print("状态码:", r.status_code)
print("返回:", r.text)

if r.status_code == 200:
    result = r.json()
    print("answer:", result.get("answer"))
    print("conversation_id:", result.get("conversation_id"))