import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from embeddings_demo.embeddings import Qwen_embeddings

index=faiss.IndexFlatL2(len(Qwen_embeddings.embed_query("test")))


vector_store=FAISS(
    embedding_function=Qwen_embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store=FAISS.load_local("vector_db", Qwen_embeddings,
                              allow_dangerous_deserialization=True)

results = vector_store.similarity_search_with_score('今天的投资建议', k=2,
                                                    filter={"date": "2026-10-31","source": "market_alert"})
for res,score in results:
    print(type(res))
    print(res.id)
    print(f"*[Score={score:3f}] {res.page_content} [{res.metadata}]")