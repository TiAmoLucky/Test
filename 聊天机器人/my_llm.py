import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# ========== 本地 Ollama 模型 ==========
local_llm = ChatOllama(
    model="deepseek-r1:7b",
    base_url="http://127.0.0.1:11434",
    temperature=0.8,
)

# ========== 云端通义千问（OpenAI 兼容模式）==========
qwen_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.8-flash",
    temperature=0.8,
)

# ========== 本地语音识别模型 ==========
model_path = r"D:\study\tools\SenseVoice\model\iic\SenseVoiceSmall"

asr_model = AutoModel(
    model=model_path,
    trust_remote_code=True,
    device="cuda:0",
)

def speech_to_text(audio_path):
    res = asr_model.generate(
        input=audio_path,
        language="auto",
        use_itn=True,
        batch_size_s=60,
    )
    return rich_transcription_postprocess(res[0]["text"])

# # ========== 测试本地 ==========
# resp = local_llm.invoke("你是谁?")
# print(resp.content)

# ========== 测试云端 ==========
# resp2 = qwen_llm.invoke("你是谁?")
# print(resp2.content)

# ========== 测试语音模型 ==========
# text = speech_to_text("测试音频路径.wav")
# print("语音识别结果:", text)