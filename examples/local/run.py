import asyncio
from litests import LiteSTS
from litests.vad.microphone_connector import start_with_pyaudio

OPENAI_API_KEY = "YOUR_API_KEY"
GOOGLE_API_KEY = "YOUR_API_KEY"

async def quick_start_main():
    sts = LiteSTS(
        vad_volume_db_threshold=-30,    # Adjust microphone sensitivity (Gate)
        stt_google_api_key=GOOGLE_API_KEY,
        llm_openai_api_key=OPENAI_API_KEY,
        cancel_echo=True,   # Set False if you want to interrupt AI's answer
        debug=True
    )

    await start_with_pyaudio(sts.vad)

asyncio.run(quick_start_main())
