import requests
import json

# 1. 配置基础信息
url = "http://127.0.0.1/v1/workflows/run"
api_key = "app-ZhwCgydA6RLWKUR8ikkYYjjK"

# 2. 设置请求头 (Headers)
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 3. 设置请求体 (Payload)
# 注意：inputs 里面的 key (这里是 "ci") 必须和 Dify 开始节点的变量名一致
payload = {
    "inputs": {
        "ci": "测试内容"
    },
    "response_mode": "blocking",  # 阻塞模式，等待全部生成完再返回
    "user": "python-test-user"
}

# 4. 发送 POST 请求
print("正在发送请求...")
response = requests.post(url, headers=headers, json=payload)

# 5. 处理返回结果
if response.status_code == 200:
    result = response.json()
    print("\n✅ 请求成功！返回结果如下：")
    # 格式化输出 JSON，方便阅读
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 通常工作流的最终输出在 data -> outputs 里面
    # outputs = result.get("data", {}).get("outputs", {})
    # print("\n提取的输出内容:", outputs)
else:
    print(f"\n❌ 请求失败，状态码: {response.status_code}")
    print(response.text)