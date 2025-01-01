import logging
from . import SpeechSynthesizer

logger = logging.getLogger(__name__)


class SpeechGatewaySpeechSynthesizer(SpeechSynthesizer):
    def __init__(
        self,
        *,
        service_name: str,
        speaker: str,
        tts_url: str = "http://127.0.0.1:8000/tts",
        audio_format: str = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        debug: bool = False
    ):
        super().__init__(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            debug=debug
        )
        self.service_name = service_name
        self.speaker = speaker
        self.tts_url = tts_url
        self.audio_format = audio_format

    async def synthesize(self, text) -> bytes:
        if not text or not text.strip():
            return bytes()

        logger.info(f"Speech synthesize: {text}")

        query_params = {"x_audio_format": self.audio_format} if self.audio_format else {}

        resp = await self.http_client.post(
            url=self.tts_url,
            params=query_params,
            json={"text": text, "service_name": self.service_name, "speaker": "0-0"}
        )

        return resp.content
