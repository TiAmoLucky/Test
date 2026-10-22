from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough, RunnableConfig
import gradio as gr

from my_llm import qwen_llm, speech_to_text

# 提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),  # 动态注入消息
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}")
])

llm = qwen_llm
chain = prompt | llm


def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat.history.db",
    )

def summarize_messages(current_input, config: RunnableConfig):
    session_id = config.get("configurable", {}).get("session_id")
    if not session_id:
        raise ValueError("必须通过config参数获取session_id")

    chat_history = get_session_history(session_id)
    stored_messages = chat_history.messages

    if len(stored_messages) <= 2:
        return {"original_messages": stored_messages, "summary": "None"}

    last_two_messages = stored_messages[-2:]
    messages_to_summarize = stored_messages[:-2]

    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "请将以下对话压缩为一条保留关键信息的摘要消息"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "请生成包含上述对话核心内容的摘要，保留重要事实和决策"),
    ])

    summarization_prompt = summarization_prompt | llm
    summarize_message = summarization_prompt.invoke({"chat_history": messages_to_summarize})

    return {
        "original_messages": last_two_messages,
        "summary": summarize_message.content
    }


chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

final_chain = (RunnablePassthrough.assign(messages_summarized=summarize_messages)) | RunnablePassthrough.assign(
    input=lambda x: x["input"],
    chat_history=lambda x: x["messages_summarized"]["original_messages"],
    # 判断逻辑修正，直接用 summary 判断
    system_message=lambda x: (
        f"你是一个乐于助人的助手。尽你所能回答所有问题。摘要:{x['messages_summarized']['summary']}"
        if x["messages_summarized"]["summary"] != "None"
        else "你是一个乐于助人的助手。尽你所能回答所有问题。"
    )
) | chain_with_message_history


# web界面中的核心函数
def add_message(chat_history, user_message):
    if user_message:
        chat_history.append({'role': 'user', 'content': user_message})
    return chat_history, ''


def execute_chain(chat_history):
    if not chat_history:
        return chat_history

    user_text = chat_history[-1]['content']

    try:
        result = final_chain.invoke(
            {'input': user_text},
            config={"configurable": {"session_id": "user123"}}
        )
        # 角色必须是 assistant，而不是 user
        chat_history.append({'role': 'assistant', 'content': result.content})
    except Exception as e:
        chat_history.append({'role': 'assistant', 'content': f"出错啦: {e}"})

    return chat_history


# 新增：语音处理函数
def process_audio(audio_path):
    if not audio_path:
        return "", gr.update()
    text = speech_to_text(audio_path)
    print(f"识别结果: {text}")
    return text, None


with gr.Blocks(title="多模态聊天机器人") as block:
    chatbot = gr.Chatbot(height=500, label="聊天机器人")
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(placeholder="请给机器人发送消息...", label="文字输入", max_lines=5)
            submit_btn = gr.Button("发送", variant="primary")

        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], label="请语音输入...", type="filepath", format="wav")
            audio_btn = gr.Button("🎤 识别并发送", variant="secondary")

    # 1. 文本提交事件（按回车）
    user_input.submit(
        add_message, [chatbot, user_input], [chatbot, user_input]
    ).then(
        execute_chain,
        [chatbot],
        [chatbot]
    )

    # 2. 按钮点击事件
    submit_btn.click(
        add_message, [chatbot, user_input], [chatbot, user_input]
    ).then(
        execute_chain,
        [chatbot],
        [chatbot]
    )

    # 3. 语音识别并发送事件
    audio_btn.click(
        process_audio,
        inputs=[audio_input],
        outputs=[user_input, audio_input]
    ).then(
        add_message,
        inputs=[chatbot, user_input],
        outputs=[chatbot, user_input]
    ).then(
        execute_chain,
        [chatbot],
        [chatbot]
    )

if __name__ == "__main__":
    block.launch(theme=gr.themes.Soft())