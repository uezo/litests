import asyncio
import io
import logging
import queue
import threading
from typing import AsyncGenerator
import wave
import pyaudio
from ..models import STSResponse
from ..pipeline import LiteSTS
from .base import Adapter

logger = logging.getLogger(__name__)


class AudioDeviceAdapter(Adapter):
    def __init__(
        self,
        sts: LiteSTS = None,
        *,
        input_sample_rate: int = 16000,
        input_channels: int = 1,
        input_chunk_size: int = 512,
        output_chunk_size: int = 1024,
        cancel_echo: bool = True
    ):
        super().__init__(sts)

        # Microphpne
        self.input_sample_rate = input_sample_rate
        self.input_channels = input_channels
        self.input_chunk_size = input_chunk_size

        # Audio player
        self.to_wave = None
        self.p = pyaudio.PyAudio()
        self.play_stream = None
        self.wave_params = None
        self.output_chunk_size = output_chunk_size

        # Echo cancellation
        self.cancel_echo = cancel_echo
        self.is_playing_locally = False
        self.sts.vad.should_mute = lambda: self.cancel_echo and self.is_playing_locally

        # Response handler
        self.stop_event = threading.Event()
        self.response_queue: queue.Queue[bytes] = queue.Queue()
        self.response_handler_thread = threading.Thread(target=self.audio_player_worker, daemon=True)
        self.response_handler_thread.start()

    # Request
    async def start_listening(self, session_id: str):
        async def start_microphone_stream() -> AsyncGenerator[bytes, None]:
            p = pyaudio.PyAudio()
            pyaudio_stream = p.open(
                rate=self.input_sample_rate,
                channels=self.input_channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.input_chunk_size
            )
            while True:
                yield pyaudio_stream.read(self.input_chunk_size)
                await asyncio.sleep(0.0001)

        await self.sts.vad.process_stream(start_microphone_stream(), session_id)

    # Response
    def audio_player_worker(self):
        while True:
            try:
                audio_data = self.response_queue.get()
                self.is_playing_locally = True
                wave_content = self.to_wave(audio_data) \
                    if self.to_wave else audio_data

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

                    data = wf.readframes(self.output_chunk_size)
                    while True:
                        data = wf.readframes(self.output_chunk_size)
                        if not data:
                            break
                        self.play_stream.write(data)

            except Exception as ex:
                logger.error(f"Error processing audio data: {ex}", exc_info=True)

            finally:
                self.is_playing_locally = False
                self.response_queue.task_done()

    async def handle_response(self, response: STSResponse):
        if response.type == "chunk" and response.audio_data:
            self.response_queue.put(response.audio_data)

    async def stop_response(self, context_id: str):
        while not self.response_queue.empty():
            try:
                _ = self.response_queue.get_nowait()
                self.response_queue.task_done()
            except:
                break

    def close(self):
        self.stop_event.set()
        self.stop_response()
        self.response_handler_thread.join()
