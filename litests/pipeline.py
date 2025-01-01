import logging
from typing import AsyncGenerator
from .models import STSRequest, STSResponse
from .vad import SpeechDetector, StandardSpeechDetector
from .stt import SpeechRecognizer
from .stt.google import GoogleSpeechRecognizer
from .llm import LLMService
from .llm.chatgpt import ChatGPTService
from .tts import SpeechSynthesizer
from .tts.voicevox import VoicevoxSpeechSynthesizer
from .response_handler import ResponseHandler
from .response_handler.playaudio import PlayWaveResponseHandler

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
        response_handler: ResponseHandler = None,
        cancel_echo: bool = False,
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
        async def on_speech_detected(data: bytes, session_id: str):
            await self.invoke(STSRequest(context_id=session_id, audio_data=data))
        self.vad.on_speech_detected = on_speech_detected

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
            system_prompt=llm_system_prompt
        )

        # Text-to-Speech
        self.tts = tts or VoicevoxSpeechSynthesizer(
            base_url=tts_voicevox_url,
            speaker=tts_voicevox_speaker,
            debug=debug
        )

        # Response handler
        self.response_handler = response_handler or PlayWaveResponseHandler()

        # Echo cancellation
        self.cancel_echo = cancel_echo
        if self.cancel_echo:
            self.vad.should_mute = lambda: self.cancel_echo and self.response_handler.is_playing_locally

    async def process_audio_samples(self, samples: bytes, context_id: str):
        await self.vad.process_samples(samples, context_id)

    async def start_with_stream(self, input_stream: AsyncGenerator[bytes, None], context_id: str):
        await self.vad.process_stream(input_stream, context_id)

    async def invoke(self, request: STSRequest):
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

        # Stop on-going response before new response
        await self.response_handler.stop_response(request.context_id)

        # LLM
        llm_stream = self.llm.chat_stream(request.context_id, recognized_text)

        # TTS
        async def synthesize_stream():
            async for llm_stream_chunk in llm_stream:
                audio_chunk = await self.tts.synthesize(llm_stream_chunk.voice_text)
                yield (audio_chunk, llm_stream_chunk.text)

        # Handle response
        await self.response_handler.handle_response(
            STSResponse(type="start", context_id=request.context_id)
        )

        response_text = ""
        response_audio = bytes()
        async for audio_chunk, text_chunk in synthesize_stream():
            response_audio += audio_chunk
            response_text += text_chunk
            # NOTE: DO NOT BROCK at response handler
            await self.response_handler.handle_response(
                STSResponse(type="chunk", context_id=request.context_id, text=text_chunk, audio_data=audio_chunk)
            )

        await self.response_handler.handle_response(
            STSResponse(type="final", context_id=request.context_id, text=response_text, audio_data=response_audio)
        )
