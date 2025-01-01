from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Awaitable, Optional

class SpeechDetector(ABC):
    def __init__(self, *, on_speech_detected: Optional[Callable[[bytes, str], Awaitable[None]]] = None,):
        self.on_speech_detected = on_speech_detected
        self.should_mute = lambda: False

    @abstractmethod
    async def process_samples(self, samples: bytes, session_id: str = None):
        pass

    @abstractmethod
    async def process_stream(self, input_stream: AsyncGenerator[bytes, None], session_id: str = None):
        pass
