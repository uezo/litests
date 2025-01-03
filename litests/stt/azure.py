import logging
from . import SpeechRecognizer

logger = logging.getLogger(__name__)


class AzureSpeechRecognizer(SpeechRecognizer):
    def __init__(
        self,
        azure_api_key: str,
        azure_region: str,
        sample_rate: int = 16000,
        language: str = "ja-JP",
        *,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        debug: bool = False
    ):
        super().__init__(
            language=language,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            debug=debug
        )
        self.azure_api_key = azure_api_key
        self.azure_region = azure_region
        self.sample_rate = sample_rate

    async def transcribe(self, data: bytes) -> str:
        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_api_key
        }

        resp = await self.http_client.post(
            f"https://{self.azure_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language={self.language}",
            headers=headers,
            content=data
        )

        try:
            resp_json = resp.json()
        except:
            resp_json = {}

        if resp.status_code != 200:
            logger.error(f"Failed in recognition: {resp.status_code}\n{resp_json}")

        if recognized_text := resp_json.get("DisplayText"):
            if self.debug:
                logger.info(f"Recognized: {recognized_text}")
            return recognized_text
