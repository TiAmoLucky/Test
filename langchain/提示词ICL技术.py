from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate

from my_llm import qwen_llm

examples = [
    {"input": "我今天很开心", "output": "[happy] I am very happy today."},
    {"input": "这让我很生气", "output": "[angry] This makes me very angry."},
    {"input": "我有点难过", "output": "[sad] I feel a little sad."},
]

example_prompt = PromptTemplate.from_template(
    "中文：{input}\n英文：{output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,  # 给模型的示例
    example_prompt=example_prompt,  # 每个示例的渲染格式
    prefix="请把下面的中文翻译成英文，并在前面加上情绪标签：",
    suffix="中文：{input}\n英文：",  # 真正要模型回答的地方
    input_variables=["input"],
)

llm = qwen_llm
chain = few_shot_prompt | llm

resp = chain.invoke({"input": "我对这件事感到非常惊讶"})
print(resp.content)
