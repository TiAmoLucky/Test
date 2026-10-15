from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core import chat_history
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langsmith.prompt_cache import configure_global_async_prompt_cache

from my_llm import local_llm

#提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个乐于助人的AI聊天助手，善于解决别人的烦恼，尽你所能回答用户的问题"),
    MessagesPlaceholder(variable_name="chat_history",optional=True),
    # ("MessagesPlaceholder","chat_history"),
    # 跟上面的写法等价
    ("human","{input}")
])

llm = local_llm

chain=prompt | llm


# 内存存储
# #存储聊天记录：内存
# store={}
#
# def get_session_history(session_id:str):
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]


def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat.history.db",
    )

#剪辑历史摘要，保留两条历史信息
def summarize_messages(current_input):
    session_id=current_input["config"]["configurable"]["session_id"]
    if not session_id:
        raise ValueError("必须通过config参数获取senssion_id")

#获取当前会话所有id历史记录
    chat_history=get_session_history(session_id)
    stored_messages = chat_history.messages

    if len(stored_messages)<=2:
        return False

    last_two_messages=stored_messages[-2:]
    messages_to_summarize = stored_messages[:-2]

    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "请将以下对话压缩为一条保留关键信息的摘要消息"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "请生成包含上述对话核心内容的摘要，保留重要事实和决策"),
    ])

    summarization_prompt= summarization_prompt | llm
    #生成摘要
    summarize_messages = summarization_prompt.invoke({"chat_history": messages_to_summarize})

    chat_history.clear()
    chat_history.add_message(summarize_messages)
    for message in last_two_messages:
        chat_history.add_message(message)

    return True

chain_with_message_history= RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

result1=chain_with_message_history.invoke({"input":"我叫吴彦祖"},config={"configurable":{"session_id":"user123"}})
print(result1.content)

result2=chain_with_message_history.invoke({"input":"我的名字是什么？"},config={"configurable":{"session_id":"user123"}})
print(result2.content)

result3=chain_with_message_history.invoke({"input":"历史上有谁和我同名同姓？"},config={"configurable":{"session_id":"user123"}})
print(result3.content)