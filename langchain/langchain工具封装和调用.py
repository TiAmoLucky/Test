import os
import requests
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

chat = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-flash",  # 确认模型名
)

city_df = pd.read_csv("city.csv")

def get_city_code(city_name: str) -> str:
    """获取城市编码"""
    match = city_df[city_df['district'] == city_name]
    if not match.empty:
        return match.iloc[0]['areacode/城市ID']
    match = city_df[city_df['city'] == city_name]
    if not match.empty:
        return match.iloc[0]['areacode/城市ID']
    match = city_df[city_df['city'].str.contains(city_name, na=False)]
    if not match.empty:
        return match.iloc[0]['areacode/城市ID']
    return 101010100

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气，返回温度及天气状况。

    Args:
        city: 城市名称，如"南京"
    """
    url = "https://eolink.o.apispace.com/456456/weather/v001/now"
    city_code = get_city_code(city)
    payload = {"areacode": city_code, "lonlat": "116.407526,39.904030"}
    headers = {"X-APISpace-Token": "pqthsgshndohzjvoa2xd1458cpfehchq"}
    response = requests.request("GET", url, params=payload, headers=headers)
    return response.text

agent = create_agent(
    model=chat,
    tools=[get_weather],
    system_prompt="你是一个天气助手，根据用户提问调用工具查询天气。",
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "南京天气怎么样"}]}
)

print(response["messages"][-1].content)