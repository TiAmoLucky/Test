import pandas as pd
import numpy as np
from embeddings_demo.embeddings import Qwen_embeddings

qwen_embedding = Qwen_embeddings


def text_2_embedding(text):
    # 返回的直接就是 list[float] 向量
    resp = qwen_embedding.embed_query(text)
    return resp


def embedding_2_file(source_file, output_file):
    """读取原始评论数据，通过调用Embedding模型，得到向量，并保存到新文件中"""
    # 步骤：1、准备数据，并读取
    df = pd.read_csv(source_file, index_col=0)

    df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]

    # 步骤2：清洗数据和 合并数据
    # 建议开启空值填充，防止报错
    df['Summary'] = df['Summary'].fillna('')
    df['Text'] = df['Text'].fillna('')

    # 把评论的摘要和内容字段合并成一个字段
    df['text_content'] = 'Summary: ' + df.Summary.str.strip() + '; Text: ' + df.Text.str.strip()

    print(f"准备处理 {len(df)} 条数据...")

    # 步骤3：调用你的模型批量生成向量
    # 注意：这里用 embed_documents，因为处理的是文档库
    print("正在生成向量，请稍候...")
    vectors = qwen_embedding.embed_documents(df['text_content'].tolist())

    # 把向量作为新列存入 DataFrame
    df['embedding'] = vectors

    # 步骤4：保存到新文件
    df.to_csv(output_file, index=False)
    print(f"✅ 处理完成！带向量的数据已保存至: {output_file}")


# ================= 3. 测试案例 =================
if __name__ == "__main__":
    # 测试 1：测试你的单条转换函数
    print("--- 测试单条文本转换 ---")
    test_text = "快递送得快不快？"
    vec = text_2_embedding(test_text)
    print(f"文本: {test_text}\n向量维度: {len(vec)}\n")

    # 测试 2：模拟语义搜索（匹配你提供的 10 条数据）
    print("--- 测试语义搜索 ---")
    reviews = [
        "物流特别快，第二天就到了",
        "快递员态度很好，还帮我送上楼",
        "衣服质量不错，面料很舒服",
        "尺码偏小，建议买大一号",
        "颜色和图片有色差，有点失望",
        "客服回复很及时，售后体验好",
        "穿了一周就起球了，质量堪忧",
        "包装很精美，送人很有面子",
        "价格比实体店便宜多了，划算",
        "洗了一次就缩水了，不推荐购买"
    ]

    # 查询向量
    query_vec = text_2_embedding("这衣服会不会坏啊？")

    # 文档向量 (实际应用中也是提前算好的)
    doc_vecs = qwen_embedding.embed_documents(reviews)

    # 计算余弦相似度
    scores = []
    for i, doc_vec in enumerate(doc_vecs):
        # numpy 计算余弦相似度
        a = np.array(query_vec)
        b = np.array(doc_vec)
        score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        scores.append((score, reviews[i]))

    # 排序并输出
    scores.sort(key=lambda x: x[0], reverse=True)

    print(f"用户查询: 「这衣服会不会坏啊？」")
    print("Top 2 匹配结果:")
    for i in range(2):
        print(f"-> [得分: {scores[i][0]:.4f}] {scores[i][1]}")