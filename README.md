# LiteSTS

A super lightweight Speech-to-Speech framework with modular VAD, STT, LLM and TTS components. 🧩

## ✨ Features

- **🧩 Modular architecture**: VAD, STT, LLM, and TTS are like building blocks—just snap them together! Each one is super simple to integrate with a lightweight interface. Here's what we support out of the box (but feel free to add your own flair!):
    - VAD: Built-in Standard VAD (turn-end detection based on silence length)
    - STT: Google, Azure and OpenAI
    - ChatGPT, Gemini, Claude. Plus, with support for LiteLLM and Dify, you can use any LLMs they support!
    - TTS: VOICEVOX / AivisSpeech, OpenAI, SpeechGateway (Yep, all the TTS supported by SpeechGateway, including Style-Bert-VITS2 and NijiVoice!)
- **🥰 Rich expression**: In addition to voice, supports text-based information exchange, enabling rich expressions like facial expressions and motions on the front end. It also supports methods like Chain-of-Thought, allowing for maximum utilization of capabilities.
- **🏎️ Super speed**: Speech synthesis and playback are performed in parallel with streaming responses from the LLM, enabling dramatically faster voice responses compared to simply connecting the components sequentially.


## 🎁 Installation

You can install it with a single `pip` command. Since `PyAudio` internally uses `PortAudio`, you'll need to install it beforehand.

```sh
pip install git+https://github.com/uezo/litests
```


## 🚀 Quick start

It's super easy to create the Speech-to-Speech AI chatbot locally:

```python
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
llm = ChatGPTService(...)

# (4) TTS
from litests.tts.voicevox import VoicevoxSpeechSynthesizer
tts = VoicevoxSpeechSynthesizer(...)

# (5) Response handler
from litests.response_handler.playaudio import PlayWaveResponseHandler
response_handler = PlayWaveResponseHandler(...)


"""
Step 2. Assign them to Speech-to-Speech pipeline
"""
sts = litests.LiteSTS(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    response_handler=response_handler,
    cancel_echo=True,
    debug=True
)


"""
Step 3. Start Speech-to-Speech with input audio data
"""
# Case 1: Microphone (PyAudio or SoundDevice)
from litests.vad.microphone_connector import start_with_pyaudio
await start_with_pyaudio(vad)

# Case 2: Generator (e.g. Audio streaming)
await sts.start_with_stream(async_generator)

# Case 3: Handled chunks (e.g. WebSocket inbound messages)
await sts.process_audio_samples(chunk, session_id)
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
    response_handler=response_handler,
    cancel_echo=True,
    debug=True
)
```


## 🧩 Make custom modules

By creating modules that inherit the interfaces for VAD, STT, LLM, TTS, and the Response Handler, you can integrate them into the pipeline. Below, only the interfaces are introduced; for implementation details, please refer to the existing modules included in the repository.


### VAD

Make the class that implements `process_samples` and `process_stream` methods.

```python
class SpeechDetector(ABC):
    @abstractmethod
    async def process_samples(self, samples: bytes, session_id: str = None):
        pass

    @abstractmethod
    async def process_stream(self, input_stream: AsyncGenerator[bytes, None], session_id: str = None):
        pass
```

### STT

Make the class that implements just `transcribe` method.

```python
class SpeechRecognizer(ABC):
    @abstractmethod
    async def transcribe(self, data: bytes) -> str:
        pass
```

### LLM

Make the class that implements `compose_messages`, `update_context` and `get_llm_stream_response` methods.

```python
class LLMService(ABC):
    @abstractmethod
    def compose_messages(self, context_id: str, text: str) -> List[dict]:
        pass

    @abstractmethod
    def update_context(self, context_id: str, request_text: str, response_text: str):
        pass

    @abstractmethod
    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[str, None]:
        pass
```

### TTS

Make the class that implements just `synthesize` method.

```python
class SpeechSynthesizer(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        pass
```

### Response Handler

Make the class that implements `handle_response` and `stop_response` methods.

```python
class ResponseHandler(ABC):
    @abstractmethod
    async def handle_response(self, response: STSResponse):
        pass

    @abstractmethod
    async def stop_response(self, context_id: str):
        pass
```


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


## 📈 Performance Recorder

Records the time taken for each component in the Speech-to-Speech pipeline, from invocation to completion.

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
