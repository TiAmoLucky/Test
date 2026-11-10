from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from deep_agent.my_llm import qwen_llm
from deep_agent.tools.tool_demo2 import calculate2

# 本地私有化部署的大模型
llm = qwen_llm
# def get_weather(city: str) -> str:  # 1个用法
#     """Get weather for a given city."""
#     return f"城市：{city}，天气很好!"

graph = create_react_agent(
    llm,
    tools=[calculate2],
    prompt="你是一个智能助手"
)