import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# ========== 本地 Ollama 模型 ==========
local_llm = ChatOllama(
    model="deepseek-r1:7b",
    base_url="http://127.0.0.1:11434",
    temperature=0.8,
)

# ========== 云端通义千问（OpenAI 兼容模式）==========
qwen_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.8-flash",
    temperature=0.8,
)


# # ========== 测试本地 ==========
# resp = local_llm.invoke("你是谁?")
# print(resp.content)

# ========== 测试云端 ==========
# resp2 = qwen_llm.invoke("你是谁?")
# print(resp2.content)