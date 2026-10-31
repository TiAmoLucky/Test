from langchain_ollama import OllamaEmbeddings

# 直接使用通过 Ollama 下载的模型
Qwen_embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# # 测试调用
# text = "测试文本"
# vector = Qwen_embeddings.embed_query(text)
# print(len(vector)) # 输出 1024