import pytest
import os
import wave
from pathlib import Path

from litests.stt.google import GoogleSpeechRecognizer

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@pytest.fixture
def stt_wav_path() -> Path:
    """
    Returns the path to the hello.wav file containing "こんにちは。"
    Make sure the file is placed at tests/data/hello.wav (or an appropriate path).
    """
    return Path(__file__).parent / "data" / "hello.wav"


@pytest.mark.asyncio
async def test_google_speech_recognizer_transcribe(stt_wav_path):
    """
    Test to verify that GoogleSpeechRecognizer can transcribe the hello.wav file
    which contains "こんにちは。".
    NOTE: This test actually calls Google's Cloud Speech-to-Text API and consumes credits.
    """
    # 1) Load the WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 2) Prepare the recognizer
    recognizer = GoogleSpeechRecognizer(
        google_api_key=GOOGLE_API_KEY,
        sample_rate=sample_rate,
        language="ja-JP",
        debug=True
    )

    # 3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 4) Check the recognized text
    assert "こんにちは" in recognized_text, f"Expected 'こんにちは', got: {recognized_text}"

    # 5) Close the recognizer's http_client
    await recognizer.close()
