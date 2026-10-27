import pandas as pd
from embeddings_demo.embeddings import Qwen_embeddings

qwen_embedding = Qwen_embeddings

def text_2_embedding(text):
    # LangChain 有现成的 embed_query 用于单条查询，不需要自己包列表了
    resp = qwen_embedding.embed_query(text)
    return resp[0]

def embedding_2_file(source_file, output_file):
    """读取原始评论数据，通过调用Embedding模型，得到向量，并保存到新文件中"""
    # 步骤：1、准备数据，并读取
    df = pd.read_csv(source_file, index_col=0)

    df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]

    print(df.head(2))

    # 步骤2：清洗数据和 合并数据
    # df = df.dropna()
    # 把评论的摘要和内容字段合并成一个字段（方便后续处理）
    df['text_content'] = 'Summary: ' + df.Summary.str.strip() + '; Text: ' + df.Text.str.strip()
    print(df.head(2))  # 增加一个text_content字段