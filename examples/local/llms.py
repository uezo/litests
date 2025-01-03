import asyncio
from litests import LiteSTS
from litests.llm.chatgpt import ChatGPTService
from litests.llm.gemini import GeminiService
from litests.llm.claude import ClaudeService
from litests.llm.dify import DifyService
from litests.llm.litellm import LiteLLMService
from litests.vad.microphone_connector import start_with_pyaudio

GOOGLE_API_KEY = "GOOGLE_API_KEY"

OPENAI_API_KEY = "OPENAI_API_KEY"
GEMINI_API_KEY = "GEMINI_API_KEY"
CLAUDE_API_KEY = "CLAUDE_API_KEY"
DIFY_API_KEY = "DIFY_API_KEY"
OTHERLLM_API_KEY = "OTHERLLM_API_KEY"


chatgpt = ChatGPTService(
    openai_api_key=OPENAI_API_KEY
)

gemini = GeminiService(
    gemini_api_key=GEMINI_API_KEY
)

claude = ClaudeService(
    anthropic_api_key=CLAUDE_API_KEY
)

dify = DifyService(
    api_key=DIFY_API_KEY,
    user="dify_user",
    base_url="your_dify_url",
    # is_agent_mode=True,   # True when type of app is agent
)

litellm = LiteLLMService(
    api_key=OTHERLLM_API_KEY,
    model="llm_service/llm_model_name",
)

async def quick_start_main():
    # Uncomment below when you use GeminiService
    # await gemini.preflight()

    sts = LiteSTS(
        vad_volume_db_threshold=-30,    # Adjust microphone sensitivity (Gate)
        stt_google_api_key=GOOGLE_API_KEY,
        llm=litellm,     # <- Select LLM service you want to use
        cancel_echo=True,
        debug=True
    )

    await start_with_pyaudio(sts.vad)

asyncio.run(quick_start_main())
