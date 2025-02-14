import asyncio
from litests import LiteSTS
from litests.stt.openai import OpenAISpeechRecognizer
from litests.tts.openai import OpenAISpeechSynthesizer
from litests.adapter.audiodevice import AudioDeviceAdapter

OPENAI_API_KEY = "YOUR_API_KEY"


async def quick_start_main():
    # STT
    stt = OpenAISpeechRecognizer(
        openai_api_key=OPENAI_API_KEY,
        alternative_languages=["en-US", "zh-CN"]
    )

    # TTS
    tts = OpenAISpeechSynthesizer(
        openai_api_key=OPENAI_API_KEY,
        speaker="shimmer",
    )

    # Create STS pipeline
    sts = LiteSTS(
        vad_volume_db_threshold=-40,    # Adjust microphone sensitivity (Gate)
        stt=stt,
        llm_openai_api_key=OPENAI_API_KEY,
        # Azure OpenAI
        # llm_model="azure",
        # llm_base_url="https://{your_resource_name}.openai.azure.com/openai/deployments/{your_deployment_name}/chat/completions?api-version={api_version}",
        tts=tts,
        debug=True
    )

    # Create adapter
    adapter = AudioDeviceAdapter(
        sts,
        cancel_echo=True    # Set False if you want to interrupt AI's answer
    )

    # Start listening
    await adapter.start_listening("_")

asyncio.run(quick_start_main())
