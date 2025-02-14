import os
import pytest

from litests.vad.standard import StandardSpeechDetector
from litests.stt.google import GoogleSpeechRecognizer
from litests.llm.chatgpt import ChatGPTService
from litests.tts.voicevox import VoicevoxSpeechSynthesizer
from litests.performance_recorder.sqlite import SQLitePerformanceRecorder
from litests.models import STSRequest, STSResponse
from litests.adapter import Adapter

from litests import LiteSTS

INPUT_VOICE_SAMPLE_RATE = 24000 # using VOICEVOX


class RecordingAdapter(Adapter):
    """
    A custom ResponseHandler that records the final audio data in memory.
    """
    def __init__(self, sts: LiteSTS):
        super().__init__(sts)
        self.final_audio = bytes()

    async def handle_response(self, response: STSResponse):
        # We only care about the "final" response which carries the entire synthesized audio
        if response.type == "final" and response.audio_data:
            self.final_audio = response.audio_data

    async def stop_response(self, context_id: str):
        # For this test, we do not need to do anything special
        pass


@pytest.mark.asyncio
async def test_lite_sts_pipeline():
    """
    Integration test scenario:
      1. Generate audio for "日本の首都は？" via Voicevox
      2. Pass that audio to LiteSTS.invoke()
         -> STT transcribes -> LLM answers (contains "東京") -> TTS re-synthesizes
      3. A custom ResponseHandler captures the final synthesized audio
      4. Use GoogleSpeechRecognizer on the final audio to check if "東京" is present.
    """
    # TTS for input audio instead of human's speech
    voicevox_for_input = VoicevoxSpeechSynthesizer(
        base_url="http://127.0.0.1:50021",
        speaker=46,
        debug=True
    )

    async def get_input_voice(text: str):
        return await voicevox_for_input.synthesize(text)

    # STT for output audio instead of human's listening
    stt_for_final = GoogleSpeechRecognizer(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        sample_rate=INPUT_VOICE_SAMPLE_RATE,
        language="ja-JP",
        debug=True
    )

    async def get_output_text(voice: bytes):
        return await stt_for_final.transcribe(voice)

    # Initialize pipeline
    lite_sts = LiteSTS(
        vad=StandardSpeechDetector(
            volume_db_threshold=-50.0,
            silence_duration_threshold=0.5,
            sample_rate=INPUT_VOICE_SAMPLE_RATE,
            debug=True
        ),
        stt=GoogleSpeechRecognizer(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            sample_rate=INPUT_VOICE_SAMPLE_RATE,
            language="ja-JP",
            debug=True
        ),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=VoicevoxSpeechSynthesizer(
            base_url="http://127.0.0.1:50021",
            speaker=46,
            debug=True
        ),
        performance_recorder=SQLitePerformanceRecorder(),  # DB記録
        debug=True
    )

    # Adapter for test
    adapter = RecordingAdapter(lite_sts)

    context_id = "test_pipeline_nippon"

    # Invoke pipeline with the first request (Ask capital of Japan)
    await lite_sts.invoke(STSRequest(context_id=context_id, audio_data=await get_input_voice("日本の首都は？")))

    # Check output voice audio
    final_audio = adapter.final_audio
    assert len(final_audio) > 0, "No final audio was captured by the response handler."
    output_text = await get_output_text(final_audio)
    assert "東京" in output_text, f"Expected '東京' in recognized text, but got: {output_text}"

    # Invoke pipeline with the successive request (Ask about of US, without using the word 'capital' to check context)
    await lite_sts.invoke(STSRequest(context_id=context_id, audio_data=await get_input_voice("アメリカは？")))

    # Check output voice audio
    final_audio = adapter.final_audio
    assert len(final_audio) > 0, "No final audio was captured by the response handler."
    output_text = await get_output_text(final_audio)
    assert "ワシントン" in output_text, f"Expected 'ワシントン' in recognized text, but got: {output_text}"

    await lite_sts.shutdown()
    await voicevox_for_input.close()
    await stt_for_final.close()
