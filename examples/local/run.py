import asyncio
from litests import LiteSTS
from litests.adapter.audiodevice import AudioDeviceAdapter

OPENAI_API_KEY = "YOUR_API_KEY"
GOOGLE_API_KEY = "YOUR_API_KEY"

# Create STS pipeline
sts = LiteSTS(
    vad_volume_db_threshold=-40,    # Adjust microphone sensitivity (Gate)
    stt_google_api_key=GOOGLE_API_KEY,
    llm_openai_api_key=OPENAI_API_KEY,
    # Azure OpenAI
    # llm_model="azure",
    # llm_base_url="https://{your_resource_name}.openai.azure.com/openai/deployments/{your_deployment_name}/chat/completions?api-version={api_version}",
    debug=True
)

# Create adapter
adapter = AudioDeviceAdapter(
    sts,
    cancel_echo=True    # Set False if you want to interrupt AI's answer
)

# Start listening
asyncio.run(adapter.start_listening("_"))
