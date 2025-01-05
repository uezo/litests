import pytest
import os
import wave
from pathlib import Path

from litests.stt.azure import AzureSpeechRecognizer

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")

@pytest.fixture
def stt_wav_path() -> Path:
    """
    Returns the path to the hello.wav file containing "こんにちは。"
    Make sure the file is placed at tests/data/hello.wav (or an appropriate path).
    """
    return Path(__file__).parent / "data" / "hello.wav"


@pytest.mark.asyncio
async def test_azure_speech_recognizer_transcribe(stt_wav_path):
    """
    Test to verify that AzureSpeechRecognizer can transcribe the hello.wav file
    which contains "こんにちは。".
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    # 1) Load the WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 2) Prepare the recognizer
    recognizer = AzureSpeechRecognizer(
        azure_api_key=AZURE_API_KEY,
        azure_region=AZURE_REGION,
        sample_rate=sample_rate,
        language="ja-JP",
        debug=True
    )

    # 3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 4) Check the recognized text
    assert "こんにちは" in recognized_text, f"Expected 'こんにちは', got: {recognized_text}"

    # 5) Invoke the transcribe_classic method
    recognized_text_classic = await recognizer.transcribe_classic(wave_data)

    # 6) Check the recognized text
    assert "こんにちは" in recognized_text_classic, f"Expected 'こんにちは', got: {recognized_text_classic}"

    # 7) Close the recognizer's http_client
    await recognizer.close()
