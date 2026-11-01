import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI
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

# ========== 云端通义千问全模态模型 (Omni) ==========
class omni_llm:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
        )
        self.model = "qwen3.5-omni-plus"
        self.voice = "Ethan"

    def chat(self, messages, use_audio=False, **kwargs):
        """统一的聊天接口"""
        # 如果不需要音频，默认只返回文本
        modalities = ["text", "audio"] if use_audio else ["text"]

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            modalities=modalities,
            audio={"voice": self.voice, "format": "wav"},
            **kwargs
        )


# 实例化一个全局对象，方便直接导入
omni = omni_llm()
