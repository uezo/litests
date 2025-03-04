from dataclasses import dataclass
from typing import List, Dict
from .llm import ToolCall


@dataclass
class STSRequest:
    type: str = "start"
    context_id: str = None
    text: str = None
    audio_data: bytes = None
    audio_duration: float = 0
    files: List[Dict[str, str]] = None


@dataclass
class STSResponse:
    type: str
    context_id: str
    text: str = None
    voice_text: str = None
    audio_data: bytes = None
    tool_call: ToolCall = None
