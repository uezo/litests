import logging
from time import time
from typing import AsyncGenerator, Tuple
from .models import STSRequest, STSResponse
from .vad import SpeechDetector, StandardSpeechDetector
from .stt import SpeechRecognizer
from .stt.google import GoogleSpeechRecognizer
from .llm import LLMService, LLMResponse
from .llm.chatgpt import ChatGPTService
from .tts import SpeechSynthesizer
from .tts.voicevox import VoicevoxSpeechSynthesizer
from .performance_recorder import PerformanceRecord, PerformanceRecorder
from .performance_recorder.sqlite import SQLitePerformanceRecorder

logger = logging.getLogger(__name__)


class LiteSTS:
    def __init__(
        self,
        *,
        vad: SpeechDetector = None,
        vad_volume_db_threshold: float = -50.0,
        vad_silence_duration_threshold: float = 0.5,
        vad_sample_rate: int = 16000,
        stt: SpeechRecognizer = None,
        stt_google_api_key: str = None,
        stt_sample_rate: int = 16000,
        llm: LLMService = None,
        llm_openai_api_key: str = None,
        llm_base_url: str = None,
        llm_model: str = "gpt-4o-mini",
        llm_system_prompt: str = None,
        tts: SpeechSynthesizer = None,
        tts_voicevox_url: str = "http://127.0.0.1:50021",
        tts_voicevox_speaker: int = 46,
        performance_recorder: PerformanceRecorder = None,
        debug: bool = False
    ):
        # Logger
        self.debug = debug
        if self.debug and not logger.hasHandlers():
            parent_logger = logging.getLogger("litests")
            parent_logger.setLevel(logging.INFO)
            log_format = logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s")
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(log_format)
            parent_logger.addHandler(streamHandler)

        # VAD
        self.vad = vad or StandardSpeechDetector(
            volume_db_threshold=vad_volume_db_threshold,
            silence_duration_threshold=vad_silence_duration_threshold,
            sample_rate=vad_sample_rate,
            debug=debug
        )

        @self.vad.on_speech_detected
        async def on_speech_detected(data: bytes, recorded_duration: float, session_id: str):
            async for response in self.invoke(STSRequest(context_id=session_id, audio_data=data, audio_duration=recorded_duration)):
                await self.handle_response(response)

        # Speech-to-Text
        self.stt = stt or GoogleSpeechRecognizer(
            google_api_key=stt_google_api_key,
            sample_rate=stt_sample_rate,
            debug=debug
        )

        # LLM
        self.llm = llm or ChatGPTService(
            openai_api_key=llm_openai_api_key,
            base_url=llm_base_url,
            model=llm_model,
            system_prompt=llm_system_prompt,
            debug=debug
        )

        # Text-to-Speech
        self.tts = tts or VoicevoxSpeechSynthesizer(
            base_url=tts_voicevox_url,
            speaker=tts_voicevox_speaker,
            debug=debug
        )

        # Response handler
        self.handle_response = self.handle_response_default
        self.stop_response = self.stop_response_default
        self._process_llm_chunk = self.process_llm_chunk_default

        # Performance recorder
        self.performance_recorder = performance_recorder or SQLitePerformanceRecorder()

    async def process_audio_samples(self, samples: bytes, context_id: str):
        await self.vad.process_samples(samples, context_id)

    async def start_with_stream(self, input_stream: AsyncGenerator[bytes, None], context_id: str):
        await self.vad.process_stream(input_stream, context_id)

    def process_llm_chunk(self, func) -> dict:
        self._process_llm_chunk = func
        return func

    async def process_llm_chunk_default(self, response: STSResponse):
        return {}

    async def handle_response_default(self, response: STSResponse):
        logger.info(f"Handle response: {response}")

    async def stop_response_default(self, context_id: str):
        logger.info(f"Stop response: {context_id}")

    async def invoke(self, request: STSRequest) -> AsyncGenerator[STSResponse, None]:
        start_time = time()
        performance = PerformanceRecord(
            request.context_id,
            stt_name=self.stt.__class__.__name__,
            llm_name=self.llm.__class__.__name__,
            tts_name=self.tts.__class__.__name__
        )

        if request.text:
            # Use text if exist
            recognized_text = request.text
            if self.debug:
                logger.info(f"Use text in request: {recognized_text}")
        elif request.audio_data:
            # Speech-to-Text
            recognized_text = await self.stt.transcribe(request.audio_data)
            if not recognized_text:
                if self.debug:
                    logger.info("No speech recognized.")
                return
            if self.debug:
                logger.info(f"Recognized text from request: {recognized_text}")
        performance.request_text = recognized_text
        performance.voice_length = request.audio_duration
        performance.stt_time = time() - start_time

        # Stop on-going response before new response
        await self.stop_response(request.context_id)
        performance.stop_response_time = time() - start_time

        # LLM
        llm_stream = self.llm.chat_stream(request.context_id, recognized_text)

        # TTS
        async def synthesize_stream() -> AsyncGenerator[Tuple[bytes, LLMResponse], None]:
            voice_text = ""
            language = None
            async for llm_stream_chunk in llm_stream:
                # LLM performance
                if performance.llm_first_chunk_time == 0:
                    performance.llm_first_chunk_time = time() - start_time

                # ToolCall
                if llm_stream_chunk.tool_call:
                    yield None, llm_stream_chunk
                    continue

                # Voice
                if llm_stream_chunk.voice_text:
                    voice_text += llm_stream_chunk.voice_text
                    if performance.llm_first_voice_chunk_time == 0:
                        performance.llm_first_voice_chunk_time = time() - start_time
                performance.llm_time = time() - start_time

                # Parse info from LLM chunk (especially, language)
                parsed_info = await self._process_llm_chunk(llm_stream_chunk)
                language = parsed_info.get("language") or language

                audio_chunk = await self.tts.synthesize(
                    text=llm_stream_chunk.voice_text,
                    style_info={"styled_text": llm_stream_chunk.text},
                    language=language
                )

                # TTS performance
                if audio_chunk:
                    if performance.tts_first_chunk_time == 0:
                        performance.tts_first_chunk_time = time() - start_time
                    performance.tts_time = time() - start_time

                yield audio_chunk, llm_stream_chunk
            performance.response_voice_text = voice_text

        # Handle response
        yield STSResponse(type="start", context_id=request.context_id)

        response_text = ""
        async for audio_chunk, llm_stream_chunk in synthesize_stream():
            if llm_stream_chunk.tool_call:
                yield STSResponse(
                    type="tool_call",
                    context_id=llm_stream_chunk.context_id,
                    tool_call=llm_stream_chunk.tool_call
                )
                continue

            response_text += llm_stream_chunk.text
            yield STSResponse(
                type="chunk",
                context_id=llm_stream_chunk.context_id,
                text=llm_stream_chunk.text,
                voice_text=llm_stream_chunk.voice_text,
                audio_data=audio_chunk
            )
        performance.response_text = response_text

        yield STSResponse(type="final", context_id=request.context_id, text=response_text)

        performance.total_time = time() - start_time
        self.performance_recorder.record(performance)

    async def finalize(self, context_id: str):
        await self.vad.finalize_session(context_id)

    async def shutdown(self):
        self.performance_recorder.close()
