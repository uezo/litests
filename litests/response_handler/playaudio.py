import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import logging
import wave
import pyaudio
from ..models import STSResponse
from . import ResponseHandlerWithQueue

logger = logging.getLogger(__name__)


class PlayWaveResponseHandler(ResponseHandlerWithQueue):
    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor()
        self.to_wave = None
        self.p = pyaudio.PyAudio()
        self.play_stream = None
        self.chunk_size = 1024

        asyncio.create_task(self.start())

    def play_audio(self, content: bytes):
        try:
            self.is_playing_locally = True

            if self.to_wave:
                wave_content = self.to_wave(content)
            else:
                wave_content = content

            with wave.open(io.BytesIO(wave_content), "rb") as wf:
                if not self.play_stream:
                    self.play_stream = self.p.open(
                        format=self.p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                    )

                data = wf.readframes(self.chunk_size)
                while True:
                    data = wf.readframes(self.chunk_size)
                    if not data:
                        break
                    self.play_stream.write(data)

        finally:
            self.is_playing_locally = False

    async def process_response_item(self, response: STSResponse):
        if response.type == "chunk":
            if response.audio_data:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.play_audio, response.audio_data)
