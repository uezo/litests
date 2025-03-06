# LiteSTS

A super lightweight Speech-to-Speech framework with modular VAD, STT, LLM and TTS components. 🧩

## ✨ Features

- **🧩 Modular architecture**: VAD, STT, LLM, and TTS are like building blocks—just snap them together! Each one is super simple to integrate with a lightweight interface. Here's what we support out of the box (but feel free to add your own flair!):
    - VAD: Built-in Standard VAD (turn-end detection based on silence length)
    - STT: Google, Azure and OpenAI
    - ChatGPT, Gemini, Claude. Plus, with support for LiteLLM and Dify, you can use any LLMs they support!
    - TTS: VOICEVOX / AivisSpeech, OpenAI, SpeechGateway (Yep, all the TTS supported by SpeechGateway, including Style-Bert-VITS2 and NijiVoice!)
- **🥰 Rich expression**: Supports text-based information exchange, enabling rich expressions like facial expressions and motions on the front end. Voice styles seamlessly align with facial expressions to enhance overall expressiveness. It also supports methods like Chain-of-Thought, allowing for maximum utilization of capabilities.
- **🏎️ Super speed**: Speech synthesis and playback are performed in parallel with streaming responses from the LLM, enabling dramatically faster voice responses compared to simply connecting the components sequentially.


## 🎁 Installation

You can install it with a single `pip` command:

```sh
pip install litests
```

If you plan to use LiteSTS to handle microphone input or play audio on a local computer, make sure to install `PortAudio` and its Python binding, `PyAudio`, beforehand:

```sh
# Mac
brew install portaudio
pip install PyAudio
```


## 🚀 Quick start

It's super easy to create the Speech-to-Speech AI chatbot locally:

```python
import asyncio
from litests import LiteSTS
from litests.adapter.audiodevice import AudioDeviceAdapter

OPENAI_API_KEY = "YOUR_API_KEY"
GOOGLE_API_KEY = "YOUR_API_KEY"

async def quick_start_main():
    sts = LiteSTS(
        vad_volume_db_threshold=-30,    # Adjust microphone sensitivity (Gate)
        stt_google_api_key=GOOGLE_API_KEY,
        llm_openai_api_key=OPENAI_API_KEY,
        # Azure OpenAI
        # llm_model="azure",
        # llm_base_url="https://{your_resource_name}.openai.azure.com/openai/deployments/{your_deployment_name}/chat/completions?api-version={api_version}",
        debug=True
    )

    adapter = AudioDeviceAdapter(sts)
    await adapter.start_listening("session_id")

asyncio.run(quick_start_main())
```

Make sure the VOICEVOX server is running at http://127.0.0.1:50021 and run the script above:

```sh
python run.py
```

Enjoy👍


## 🛠️ Customize pipeline

By instantiating modules for VAD, STT, LLM, TTS, and the Response Handler, and assigning them in the LiteSTS pipeline constructor, you can fully customize the features of the pipeline.


```python
"""
Step 1. Create modular components
"""
# (1) VAD
from litests.vad import StandardSpeechDetector
vad = StandardSpeechDetector(...)

# (2) STT
from litests.stt.google import GoogleSpeechRecognizer
stt = GoogleSpeechRecognizer(...)

# (3) LLM
from litests.llm.chatgpt import ChatGPTService
llm = ChatGPTService(
    ...
    context_manager=PostgreSQLContextManager(...)   # <- Set if you use PostgreSQL
)

# (4) TTS
from litests.tts.voicevox import VoicevoxSpeechSynthesizer
tts = VoicevoxSpeechSynthesizer(...)

# (5) Performance Recorder
from litests.performance_recorder.postgres import PostgreSQLPerformanceRecorder
performance_recorder = PostgreSQLPerformanceRecorder(...)


"""
Step 2. Assign them to Speech-to-Speech pipeline
"""
sts = litests.LiteSTS(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    performance_recorder=performance_recorder,
    debug=True
)


"""
Step 3. Start Speech-to-Speech with adapter
"""
# Case 1: Microphone (PyAudio)
adapter = AudioDeviceAdapter(sts)
await adapter.start_listening("session_id")

# Case 2: WebSocket (Twilio)
class TwilioAdapter(WebSocketAdapter):
    # Implement adapter for twilio
    ...

adapter = TwilioAdapter(sts)
router = adapter.get_websocket_router(wss_base_url="wss://your_domain")
app = FastAPI()
app.include_router(router)
tts.audio_format = "mulaw"  # <- TTS service should support mulaw
```

See also `examples/local/llms.py`. For example, you can use Gemini by the following code:

```python
gemini = GeminiService(
    gemini_api_key=GEMINI_API_KEY
)

sts = litests.LiteSTS(
    vad=vad,
    stt=stt,
    llm=gemini,     # <- Set gemini here
    tts=tts,
    debug=True
)
```


## ⚡️ Function Calling

You can use Function Calling (Tool Call) by registering function specifications and their handlers through `tool` decorator, as shown below. Functions will be automatically invoked as needed.

**NOTE**: Currently, only ChatGPT is supported.

```python
# Create LLM service
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    system_prompt=SYSTEM_PROMPT
)

# Register tool
weather_tool_spec = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
        },
    }
}
@llm.tool(weather_tool_spec)    # NOTE: Gemini doesn't take spec as argument
async def get_weather(location: str = None):
    weather = await weather_api(location=location)
    return weather  # {"weather": "clear", "temperature": 23.4}
```


## ⛓️ Chain of Thought Prompting

Chain of Thought Prompting (CoT) is one of the popular techniques to improve the quality of AI responses. LiteSTS, by default, directly synthesize AI output, but it can also be configured to synthesize only the text inside specific XML tags.

For example, if you want the AI to output its thought process inside `<thinking>~</thinking>` and the final speech content inside `<answer>~</answer>`, you can use the following sample code:

```python
SYSTEM_PROMPT = """
Carefully consider the response first.
Output your thought process inside <thinking>~</thinking>.
Then, output the content to be spoken inside <answer>~</answer>.
"""

service = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    system_prompt=SYSTEM_PROMPT,
    model="gpt-4o",
    temperature=0.5,
    voice_text_tag="answer" # <- Synthesize inner text of <answer> tag
)
```


## 🪄 Request Filter

You can validate and preprocess requests (recognized text from voice) before they are sent to LLM.

```python
# Create LLM service
chatgpt = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    system_prompt=SYSTEM_PROMPT,
    debug = True
)

# Set filter
@chatgpt.request_filter
def request_filter(text: str):
    return f"Here is the user's spoken input. Respond as if you're a cat, adding 'meow' to the end of each sentence.\n\nUser: {text}"
```

**System Prompt vs Request Filter:** While the system prompt is generally static and used to define overall behavior, the request filter can dynamically insert instructions based on the specific context. It can emphasize key points to prioritize in generating responses, helping stabilize the conversation and adapt to changing scenarios.


## 🌏 Multi-language Support

You can dynamically switch the spoken language during a conversation.  
To enable this, configure the system prompt, set up `SpeechSynthesizer`, and add custom logic to the Speech-to-Speech pipeline as shown below:

```python
# System prompt
SYSTEM_PROMPT = """
You can speak following languages:

- English (en-US)
- Chinese (zh-CN)

When responding in a language other than Japanese, always prepend `[lang:en-US]` or `[lang:zh-CN]` at the beginning of the response.  
Additionally, when switching back to Japanese from another language, always prepend `[lang:ja-JP]` at the beginning of the response.
"""

# Setup TTS and configure language-speaker map
tts = GoogleSpeechSynthesizer(
    google_api_key=GOOGLE_API_KEY,
    speaker="ja-JP-Standard-B",
)
tts.voice_map["en-US"] = "en-US-Standard-H"     # English
tts.voice_map["cmn-CN"] = "cmn-CN-Standard-D"   # Chinese

# Add parsing logic for language code
import re
@sts.process_llm_chunk
async def process_llm_chunk(chunk: STSResponse):
    match = re.search(r"\[lang:([a-zA-Z-]+)\]", chunk.text)
    if match:
        return {"language": match.group(1)}
    else:
        return {}
```


## 🥰 Voice Style

You can apply a specific voice style to synthesized speech when certain keywords are included in the response.

To use this feature, register the keywords and their corresponding speaker names or styles for each TTS component in a `style_mapper`.

```python
# VOICEVOX / AivisSpeech
from litests.tts.voicevox import VoicevoxSpeechSynthesizer
voicevox_tts = VoicevoxSpeechSynthesizer(
    # AivisSpeech
    base_url="http://127.0.0.1:10101",
    # Base speaker name for the neutral style (Anneli / Neutral)
    speaker=888753761,
    # Define style mapper (Keyword in response : styled speaker)
    style_mapper={
        "[face:Joy]": "888753764",
        "[face:Angry]": "888753765",
        "[face:Sorrow]": "888753765",
        "[face:Fun]": "888753762",
        "[face:Surprised]": "888753762"
    },
    debug=True
)

# SpeechGateway
from litests.tts.speech_gateway import SpeechGatewaySpeechSynthesizer
tts = SpeechGatewaySpeechSynthesizer(
    tts_url="http://127.0.0.1:8000/tts",
    service_name="sbv2",
    speaker="0-0",
    # Define style mapper (Keyword in response : voice style)
    style_mapper = {
        "[face:Joy]": "joy",
        "[face:Angry]": "angry",
        "[face:Sorrow]": "sorrow",
        "[face:Fun]": "fun",
        "[face:Surprised]": "surprised",
    },
    debug=True
)
```


## 💾 Long-term Memory

LiteSTS doesn't have long-term memory features itself but can work with libraries for long-term memory like mem0, zep, [ChatMemory](https://github.com/uezo/chatmemory) etc.

Here is an example for ChatMemory.

```python
import httpx

@sts.on_finish
async def on_finish(request: STSRequest, response: STSResponse):
    if not not request.context_id or not request.text or not response.voice_text:
        return

    # Send history to ChatMemory service (We recommend async call)
    httpx.post(
        url=f"http://127.0.0.1:8000/history",   # ChatMemory API
        json={
            "user_id": request.user_id or "litests",
            "session_id": request.context_id,
            "messages": [
                {"role": "user", "content": request.text},
                {"role": "assistant", "content": response.voice_text}
            ]
        }
    )
```

To retrieve the memory, register calling `/search` as a tool.


## 🔌 WebSocket

Refer to `examples/websocket`.

- server.py: A WebSocket server program. Set your API keys and start this first.

    ```sh
    uvicorn server:app
    ```

- client.py: A WebSocket client program. Run this after starting the server, and start a conversation by saying something.

    ```sh
    python client.py
    ```

**NOTE**: To make the core mechanism easier to understand, exception handling and resource cleanup have been omitted. If you plan to use this in a production service, be sure to implement these as well.


## 📈 Performance Recorder and Voice Recorder

Performance Recorder records the time taken for each component in the Speech-to-Speech pipeline, from invocation to completion.

The recorded metrics include:

- `stt_time`: Time taken for transcription by the Speech-to-Text service.
- `stop_response_time`: Time taken to stop the response of a previous request, if any.
- `llm_first_chunk_time`: Time taken to receive the first sentence from the LLM.
- `llm_first_voice_chunk_time`: Time taken to receive the first sentence from the LLM that is used for speech synthesis.
- `llm_time`: Time taken to receive the full response from the LLM.
- `tts_first_chunk_time`: Time taken to synthesize the first sentence for speech synthesis.
- `tts_time`: Time taken to complete the entire speech synthesis process.
- `total_time`: Total time taken for the entire pipeline to complete.

The key metric is `tts_first_chunk_time`, which measures the time between when the user finishes speaking and when the system begins its response.

By default, SQLite is used for storing data, but you can implement a custom recorder by using the `PerformanceRecorder` interface.

Voice Recorder records the audio data of request and response voices. This feature allows you to check what audio was recognized and whether the synthesized speech was pronounced correctly.

By default, the audio data is stored in the file system. We also provide the one that stores data to Azure Blob Storage. You can disable voice recorder by setting `voice_recorder_enabled=True` to LiteSTS pipeline instance.
