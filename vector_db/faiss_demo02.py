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

# 2. 准备数据 (Document)
document_1 = Document(
    page_content="今天早晨我吃了巧克力薄煎饼和炒蛋。",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="明天的天气预报是阴天多云，最高气温62华氏度。",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="正在用LangChain构建一个激动人心的新项目—快来看看吧！",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="今日A股三大指数集体高开，AI算力板块领涨，建议关注科技龙头的长期配置价值。",
    metadata={"source": "finance_news", "date": "2026-10-31", "author": "分析师张三"}
)

document_5 = Document(
    page_content="近期黄金价格震荡回调，短期存在回调风险，但长线看多逻辑未变，建议逢低分批建仓。",
    metadata={"source": "finance_report", "date": "2026-10-31", "author": "分析师李四"}
)

document_6 = Document(
    page_content="本周五将发布最新的非农就业数据，请投资者密切关注，注意控制仓位防范隔夜风险。",
    metadata={"source": "market_alert", "date": "2026-10-31", "author": "宏观研究员赵六"}
)
document_7 = Document(
    page_content="近期黄金价格震荡回调，短期存在回调风险，但长线看多逻辑未变，建议逢低分批建仓。",
    metadata={"source": "finance_report", "date": "2026-10-31", "author": "分析师李四"}
)

document_8 = Document(
    page_content="对于稳健型投资者，建议将70%的资金配置于固收类产品，30%用于指数基金定投，以降低波动风险。",
    metadata={"source": "investment_guide", "date": "2026-10-30", "author": "理财规划师王五"}
)

document_9 = Document(
    page_content="本周五将发布最新的非农就业数据，请投资者密切关注，注意控制仓位防范隔夜风险。",
    metadata={"source": "market_alert", "date": "2026-10-31", "author": "宏观研究员赵六"}
)

documents=[
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
]

ids = ['id'+str(i+1) for i in range(len(documents))]

vector_store.add_documents(documents, ids=ids)

vector_store.save_local("vector_db")