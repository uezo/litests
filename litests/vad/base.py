from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Awaitable, Optional

class SpeechDetector(ABC):
    def __init__(self, *, sample_rate: int = 16000, on_speech_detected: Optional[Callable[[bytes, float, str], Awaitable[None]]] = None,):
        self.sample_rate = sample_rate
        self.on_speech_detected = on_speech_detected
        self.should_mute = lambda: False

    @abstractmethod
    async def process_samples(self, samples: bytes, session_id: str = None):
        pass

    @abstractmethod
    async def process_stream(self, input_stream: AsyncGenerator[bytes, None], session_id: str = None):
        pass
