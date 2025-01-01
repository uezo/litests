from logging import getLogger
from typing import AsyncGenerator, Dict, List
import openai
from . import LLMService

logger = getLogger(__name__)


class ChatGPTService(LLMService):
    def __init__(
        self,
        *,
        openai_api_key: str = None,
        system_prompt: str = None,
        base_url: str = None,
        model: str = "gpt-4o",
        temperature: float = 0.5,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        skip_before: str = None
    ):
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            split_chars=split_chars,
            option_split_chars=option_split_chars,
            option_split_threshold=option_split_threshold,
            skip_before=skip_before
        )
        self.openai_client = openai.AsyncClient(api_key=openai_api_key, base_url=base_url)
        self.contexts: List[Dict[str, List]] = {}

    def compose_messages(self, context_id: str, text: str) -> List[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.contexts.get(context_id, []))
        messages.append({"role": "user", "content": text})
        return messages

    def update_context(self, context_id: str, request_text: str, response_text: str):
        messages = self.contexts.get(context_id, [])
        messages.append({"role": "user", "content": request_text})
        messages.append({"role": "assistant", "content": response_text})
        self.contexts[context_id] = messages

    async def get_llm_stream_response(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        stream_resp = await self.openai_client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            stream=True
        )

        response_text = ""
        async for chunk in stream_resp:
            if content := chunk.choices[0].delta.content:
                response_text += content
                yield content
