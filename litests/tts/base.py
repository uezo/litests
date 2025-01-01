from abc import ABC, abstractmethod
import httpx
import logging

logger = logging.getLogger(__name__)


class SpeechSynthesizer(ABC):
    def __init__(
        self,
        *,
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

        self.debug = debug

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        pass

    async def close(self):
        await self.http_client.aclose()
