import pandas as pd
import numpy as np
import ast
from embeddings_demo.embeddings import Qwen_embeddings

qwen_embedding = Qwen_embeddings

def text_2_embedding(text):
    # LangChain 有现成的 embed_query 用于单条查询，不需要自己包列表了
    return qwen_embedding.embed_query(text)


def embedding_2_file(source_file, output_file):

    # 步骤1：准备数据，并读取
    df = pd.read_csv(
        source_file,
        header=None,  #告诉 Pandas 第一行不是表头
        usecols=[0],  # 只读取第一列
        names=["text_content"]  # 给字段命名
    )

    print(df.head(3))

    # 步骤2：清洗数据
    df = df.dropna(subset=["text_content"])

    # 步骤3：调用 Embedding 模型，得到向量
    df["embedding"] = df["text_content"].apply(text_2_embedding)

    df.to_csv(output_file, index=False)


def cosine_distance(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_text(input, embedding_file, top_n=3):
    """
    根据用户输入的问题，进行语义检索，返回最相似的前top_n个结果
    :param input:
    :param top_n:
    :return:
    """
    df_data = pd.read_csv(embedding_file)
    # 把字符串变成向量，保存到新字段
    df_data['embedding_vector'] = df_data['embedding'].apply(ast.literal_eval)

    input_vector = text_2_embedding(input)

    df_data['similarity'] = df_data.embedding_vector.apply( lambda x: cosine_distance(x, input_vector) )

    res = (
        df_data.sort_values('similarity', ascending=False)
        .head(top_n)
        .text_content.str.replace('Summary: ', '')  # text_content是字段名
        .str.replace('; Text: ', ';')
    )
    df_data = pd.read_csv(embedding_file)

    for r in res:
        print(r)
        print("-"*30)

if __name__ == '__main__':
    # embedding_2_file(source_file="test.csv",output_file="output_test.csv")
    search_text("衣服质量怎么样", "output_test.csv")
