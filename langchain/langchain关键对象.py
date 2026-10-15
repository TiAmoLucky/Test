from importlib.resources import contents

from langchain_classic.chains import LLMChain
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

chat = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.8-flash"
)


response_schema = [
    ResponseSchema(name="answer",description="问题答案",type="string"),
    ResponseSchema(name="confidence",description="答案的置信度",type="float")
    ]
output_parser = StructuredOutputParser.from_response_schemas(response_schema)

template = """你是一名数学老师，请用{style}风格回答下列问题，并以JOSN格式返回问题答案和置信度：
问题：{question}
{format_instructions}"""

prompt=PromptTemplate(
    template=template,
partial_variables={"format_instructions":output_parser.get_format_instructions()}
)

llm_chain = prompt|chat

chunks=[]

for chunk in llm_chain.stream({"style":"通俗易懂","question":"勾股定理是什么？"}):
    chunks.append(chunk)
    print(chunk.content,end="",flush=True)

