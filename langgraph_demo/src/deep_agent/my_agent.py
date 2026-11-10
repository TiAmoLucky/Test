from langgraph.prebuilt import create_react_agent

from deep_agent.tools.tool_demo2 import calculate2
from my_llm import local_qwen

llm=local_qwen

# def get_weather(city: str) -> str:  # 1个用法
#     """Get weather for a given city."""
#     return f"城市：{city}，天气很好!"

graph = create_react_agent(
    llm,
    tools=[calculate2],
    prompt="你是一个智能助手"
)

result=graph.stream(
    input={"messages": [{"role":"user","content":"(9+33)*7/2"}]},
    stream_mode="values"
)

for chunk in result:
    print(chunk["messages"][-1].content)




