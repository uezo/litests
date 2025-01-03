from dataclasses import dataclass


@dataclass
class STSRequest:
    type: str = "start"
    context_id: str = None
    text: str = None
    audio_data: bytes = None
    audio_duration: float = 0


@dataclass
class STSResponse:
    type: str
    context_id: str
    text: str = None
    audio_data: bytes = None
