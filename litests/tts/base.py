from abc import ABC, abstractmethod
from typing import Dict
import httpx
import logging

logger = logging.getLogger(__name__)


class SpeechSynthesizer(ABC):
    def __init__(
        self,
        *,
        style_mapper: Dict[str, str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        debug: bool = False
    ):
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )
        self.style_mapper = style_mapper or {}
        self.debug = debug

    def parse_style(self, style_info: dict = None) -> str:
        if not style_info:
            return None

        styled_text = style_info.get("styled_text", "")
        for k, v in self.style_mapper.items():
            if k in styled_text:
                return v
        return None

    @abstractmethod
    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        pass

    async def close(self):
        await self.http_client.aclose()
