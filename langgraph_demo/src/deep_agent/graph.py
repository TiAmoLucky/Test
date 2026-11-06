from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# 本地私有化部署的大模型
llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0.8,
    base_url="http://127.0.0.1:11434",
    model_kwargs={'chat_template_kwargs': {'enable_thinking': False}},
)

def get_weather(city: str) -> str:  # 1个用法
    """Get weather for a given city."""
    return f"城市：{city}，天气很好!"

graph = create_react_agent(
    llm,
    tools=[get_weather],
    prompt="你是一个智能助手"
)