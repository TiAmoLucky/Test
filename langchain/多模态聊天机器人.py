from email.mime import audio

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
import gradio as gr

from my_llm import local_llm

#提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system","{system_message}"),  #动态注入消息
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
        return {"original_messages":stored_messages,"summary":"None"}

    last_two_messages=stored_messages[-2:]
    messages_to_summarize = stored_messages[:-2]

    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "请将以下对话压缩为一条保留关键信息的摘要消息"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "请生成包含上述对话核心内容的摘要，保留重要事实和决策"),
    ])

    summarization_prompt= summarization_prompt | llm
    #生成摘要
    summarize_message = summarization_prompt.invoke({"chat_history": messages_to_summarize})

    return {
        "original_messages":last_two_messages,
        "summary":summarize_message
    }

    return True

chain_with_message_history= RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

final_chain = (RunnablePassthrough.assign(messages_summarized=summarize_messages)) | RunnablePassthrough.assign(
    input=lambda x: x["input"],
    chat_history=lambda x: x["messages_summarized"]["original_messages"],
    system_message=lambda x: f"你是一个乐于助人的助手。尽你所能回答所有问题。摘要:{x['messages_summarized']['summary']}"
    if x["messages_summarized"].get("summary.content") else "无摘要"
) | chain_with_message_history


# result1=final_chain.invoke({"input":"你好，我的名字叫吴彦祖","config":{"configurable":{"session_id":"user123"}}},
#                                           config={"configurable": {"session_id": "user123"}})
# print(result1.content)
#
# result2=final_chain.invoke({"input":"请你用我的名字写一首诗？","config":{"configurable":{"session_id":"user123"}}},
#                                           config={"configurable":{"session_id":"user123"}})
# print(result2.content)
#
# result3=final_chain.invoke({"input":"请你将这首诗进行润色，需要对仗工整，五言绝句","config":{"configurable":{"session_id":"user123"}}},
#                                           config={"configurable":{"session_id":"user123"}})
# print(result3.content)

with gr.Blocks(title="多模态聊天机器人",theme=gr.themes.soft()) as block:
    chatbot=gr.Chatbot(type="messages",height=500,label="聊天机器人")
    with gr.Row:
        with gr.Column(scale=4):
            uesr_input=gr.Textbox(placeholder="请给机器人发送消息...",label="文字输入",max_lines=5)
            submit_btn=gr.Button("发送",variant="primary")

        with gr.Column(scale=1):
            audio_input=gr.Audio(sources=["macrophone"],lable="请语音输入...",type="filepath",format="wav")

if __name__=="__main__":
    block.launch()