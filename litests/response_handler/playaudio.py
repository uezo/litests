import io
import logging
import wave
from ..models import STSResponse
from . import ResponseHandlerWithQueue

logger = logging.getLogger(__name__)


class PlayWaveResponseHandler(ResponseHandlerWithQueue):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.p = None

        try:
            import pyaudio

            self.to_wave = None
            self.p = pyaudio.PyAudio()
            self.play_stream = None
            self.wave_params = None
            self.chunk_size = 1024

        except Exception as ex:
            logger.warning(f"Error at __init__ in PlayWaveResponseHandler: {ex}")
            logger.warning("Response handler will just print responses.")

    def play_audio(self, content: bytes):
        try:
            self.is_playing_locally = True

            if self.to_wave:
                wave_content = self.to_wave(content)
            else:
                wave_content = content

            with wave.open(io.BytesIO(wave_content), "rb") as wf:
                current_params = wf.getparams()
                if not self.play_stream or self.wave_params != current_params:
                    self.wave_params = current_params
                    self.play_stream = self.p.open(
                        format=self.p.get_format_from_width(self.wave_params.sampwidth),
                        channels=self.wave_params.nchannels,
                        rate=self.wave_params.framerate,
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
            if self.p:
                # Voice mode
                if response.audio_data:
                    self.play_audio(response.audio_data)
            else:
                # Text mode
                if response.type == "chunk":
                    print(f"AI: {response.text}")
                elif response.type == "final":
                    print(f"context_id={response.context_id}, type={response.type}, audio_data={len(response.audio_data or [])}bytes")
