from langchain_core.prompts import PromptTemplate

from 多模态聊天机器人.my_llm import qwen_llm

prompt_template = PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的报幕词。")
res = prompt_template.invoke({"topic": "相声"})

llm = qwen_llm

chain = prompt_template | llm
response = chain.invoke({"topic": "相声"})
print(response)
