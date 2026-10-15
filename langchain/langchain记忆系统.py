import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.8-flash"
)


