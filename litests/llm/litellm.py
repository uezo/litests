from logging import getLogger
from typing import AsyncGenerator, Dict, List
from litellm import acompletion
from . import LLMService

logger = getLogger(__name__)


class LiteLLMService(LLMService):
    def __init__(
        self,
        *,
        api_key: str = None,
        system_prompt: str = None,
        system_prompt_by_user_prompt: bool = False,
        base_url: str = None,
        model: str = None,
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
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt_by_user_prompt = system_prompt_by_user_prompt
        self.contexts: List[Dict[str, List]] = {}

    def compose_messages(self, context_id: str, text: str) -> List[dict]:
        messages = []
        if self.system_prompt:
            if self.system_prompt_by_user_prompt:
                messages.append({"role": "user", "content": self.system_prompt})
                messages.append({"role": "assistant", "content": "ok"})
            else:
                messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.contexts.get(context_id, []))
        messages.append({"role": "user", "content": text})
        return messages

    def update_context(self, context_id: str, request_text: str, response_text: str):
        messages = self.contexts.get(context_id, [])
        messages.append({"role": "user", "content": request_text})
        messages.append({"role": "assistant", "content": response_text})
        self.contexts[context_id] = messages

    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[str, None]:
        stream_resp = await acompletion(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True
        )

        async for chunk in stream_resp:
            if content := chunk.choices[0].delta.content:
                yield content
