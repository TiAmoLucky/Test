from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
import gradio as gr

from my_llm import qwen_llm

#提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system","{system_message}"),  #动态注入消息
    MessagesPlaceholder(variable_name="chat_history",optional=True),
    # ("MessagesPlaceholder","chat_history"),
    # 跟上面的写法等价
    ("human","{input}")
])

llm = qwen_llm

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

chain_with_message_history= RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

final_chain = (RunnablePassthrough.assign(messages_summarized=summarize_messages)) | RunnablePassthrough.assign(
    input=lambda x: x["input"],
    chat_history=lambda x: x["messages_summarized"]["original_messages"],
    system_message=lambda x: (
        f"你是一个乐于助人的助手。尽你所能回答所有问题。摘要:{x['messages_summarized']['summary']}"
        if x["messages_summarized"]["summary"] != "None"
        else "你是一个乐于助人的助手。尽你所能回答所有问题。"
    )
) | chain_with_message_history

def get_last_user_after_assistant(history):
    """反向遍历找到最后一个assistant的位置，并返回后面的所有user消息"""
    if not history:
        return None
    if history[-1]["role"] == "assistant":
        return None

    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    # 如果没有找到assistant
    if last_assistant_idx == -1:
        return history
    else:
        # 从assistant位置向后查找第一个user
        return history[last_assistant_idx+1:]


def add_message(history, messages):
    for m in messages["files"]:
        print(m)
        history.append({"role": "user", "content": {"path":m}})

    if messages["text"] is None:
        history.append({"role": "user", "content": messages["text"]})
    return history,gr.MultimodalTextbox(value=None,interactive=False)

def submit_messages(history):
    user_messages=get_last_user_after_assistant(history)
    print(user_messages)

with gr.Blocks(title="多模态聊天机器人") as block:
    chatbot=gr.Chatbot(height=500,label="聊天机器人")
    # 创建多模态输入框
    chat_input = gr.MultimodalTextbox(
        interactive=True,  # 可交互
        file_types=['image', '.wav', '.mp4'],
        file_count="multiple",  # 允许多文件上传
        placeholder="请输入信息或者上传文件...",  # 输入框提示文本
        show_label=False,  # 不显示标签
        sources=["microphone", "upload"],  # 支持麦克风和文件上传
    )

    chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    ).then(
        submit_messages,
        [chatbot],
        [chatbot],
    )


if __name__=="__main__":
    block.launch(theme=gr.themes.Soft())