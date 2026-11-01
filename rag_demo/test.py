import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    requests_kwargs={
        "headers": {"User-Agent": os.environ["USER_AGENT"]},
        "timeout": 10  # 将 timeout 放到 requests_kwargs 里
    }
)

docs = loader.load()
print(f"加载文档数：{len(docs)}")
if docs:
    print(docs[0].page_content[:300])