import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# ========== 本地 Ollama 模型 ==========
local_deepseek = ChatOllama(
    model="deepseek-r1:7b",
    base_url="http://127.0.0.1:11434",
    temperature=0.8,
)

local_qwen = ChatOllama(
    model="qwen3.5:4b",
    temperature=0.8,
    base_url="http://127.0.0.1:11434",
    model_kwargs={'chat_template_kwargs': {'enable_thinking': False}},
)

# ========== 云端通义千问（OpenAI 兼容模式）==========
qwen_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.8-flash",
    temperature=0.8,
)
