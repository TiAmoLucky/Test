import uuid
import os
import base64
import gradio as gr
import io

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from PIL import Image


# 1. 初始化千问模型
omni = ChatOpenAI(
    model="qwen3.5-omni-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    model_kwargs={
        "modalities": ["text"]
    }
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个全模态AI助手，可以处理音频、文本和图像。"
                   "当用户发送音频时，请理解音频内容并直接回答用户的问题。"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | omni


def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat.history.db",
    )


chain_history = RunnableWithMessageHistory(
    chain,
    get_session_history
)

config = {
    "configurable": {
        "session_id": str(uuid.uuid4())
    }
}


def transcribe_audio(audio_path):
    """把本地音频转换为千问需要的Base64输入格式"""
    try:
        with open(audio_path, "rb") as audio_file:
            audio_data = base64.b64encode(
                audio_file.read()
            ).decode("utf-8")

        return {
            "type": "input_audio",
            "input_audio": {
                "data": f"data:;base64,{audio_data}",
                "format": "wav"
            }
        }

    except Exception as e:
        print(f"错误：{e}")
        return {}


def transcribe_image(image_path):
    """把本地图片转换为Base64输入格式"""
    with Image.open(image_path) as img:
        img_format = img.format if img.format else "JPEG"

        buffered = io.BytesIO()
        img.save(buffered, format=img_format)

        image_data = base64.b64encode(
            buffered.getvalue()
        ).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": (
                    f"data:image/{img_format.lower()};"
                    f"base64,{image_data}"
                ),
                "detail": "low"
            }
        }


def get_last_user_after_assistant(history):
    """返回最后一条助手消息之后的所有用户消息"""
    if not history:
        return None

    if history[-1]["role"] == "assistant":
        return None

    last_assistant_idx = -1

    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx == -1:
        return history

    return history[last_assistant_idx + 1:]


def add_message(history, messages):
    history = history or []

    for file_item in messages.get("files", []):
        if isinstance(file_item, dict):
            file_path = file_item.get("path")
        else:
            file_path = getattr(file_item, "path", str(file_item))

        history.append({
            "role": "user",
            "content": {
                "path": file_path
            }
        })

    if messages.get("text"):
        history.append({
            "role": "user",
            "content": messages["text"]
        })

    return (
        history,
        gr.MultimodalTextbox(
            value=None,
            interactive=False
        )
    )


def submit_messages(history):
    user_messages = get_last_user_after_assistant(history)
    content = []

    if user_messages:
        for message in user_messages:
            message_content = message.get("content")
            if isinstance(message_content, list):
                for item in message_content:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        if text.strip():
                            content.append({
                                "type": "text",
                                "text": text
                            })
                    elif item.get("type") == "file":
                        file_data = item.get("file", {})
                        file_path = file_data.get("path")

                        if not file_path:
                            continue

                        file_path_lower = file_path.lower()

                        if file_path_lower.endswith(
                            (".wav", ".mp3", ".m4a", ".aac", ".ogg")
                        ):
                            file_message = transcribe_audio(file_path)

                            if file_message:
                                content.append(file_message)

                        elif file_path_lower.endswith(
                            (".jpg", ".jpeg", ".png")
                        ):
                            content.append(
                                transcribe_image(file_path)
                            )

            elif isinstance(message_content, str):
                if message_content.strip():
                    content.append({
                        "type": "text",
                        "text": message_content
                    })

    if not content:
        history.append({
            "role": "assistant",
            "content": "没有检测到有效的输入内容。"
        })
        return history

    input_message = HumanMessage(content=content)

    response = chain_history.invoke({"messages": [input_message]},config=config)

    history.append({
        "role": "assistant",
        "content": response.content
    })

    return history

with gr.Blocks(title="聊天机器人") as block:
    chatbot = gr.Chatbot(
        height=500,
        label="聊天机器人",
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image", ".wav"],
        file_count="multiple",
        placeholder="请输入信息或者上传文件……",
        show_label=False,
        sources=["microphone", "upload"],
    )

    chat_input.submit(
        add_message,
        [chatbot, chat_input],
        [chatbot, chat_input]
    ).then(
        submit_messages,
        [chatbot],
        [chatbot]
    ).then(
        lambda: gr.update(
            interactive=True,
            value=None
        ),
        None,
        [chat_input]
    )


if __name__ == "__main__":
    block.launch(theme=gr.themes.Soft())