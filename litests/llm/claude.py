from logging import getLogger
from typing import AsyncGenerator, Dict, List
from anthropic import AsyncAnthropic
from . import LLMService

logger = getLogger(__name__)


class ClaudeService(LLMService):
    def __init__(
        self,
        *,
        anthropic_api_key: str = None,
        system_prompt: str = None,
        base_url: str = None,
        model: str = "claude-3-5-sonnet-latest",
        temperature: float = 0.5,
        max_tokens: int = 1024,
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
        self.anthropic_client = AsyncAnthropic(
            api_key=anthropic_api_key,
            base_url=base_url
        )
        self.max_tokens = max_tokens
        self.contexts: List[Dict[str, List]] = {}

    def compose_messages(self, context_id: str, text: str) -> List[dict]:
        messages = []
        messages.extend(self.contexts.get(context_id, []))
        messages.append({"role": "user", "content": text})
        return messages

    def update_context(self, context_id: str, request_text: str, response_text: str):
        messages = self.contexts.get(context_id, [])
        messages.append({"role": "user", "content": request_text})
        messages.append({"role": "assistant", "content": response_text})
        self.contexts[context_id] = messages

    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[str, None]:
        async with self.anthropic_client.messages.stream(
            messages=messages,
            system=self.system_prompt or "",
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ) as stream_resp:
            async for chunk in stream_resp.text_stream:
                if content := chunk:
                    yield content
