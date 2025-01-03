import asyncio
from logging import getLogger
from typing import AsyncGenerator, Dict, List
import google.generativeai as genai
from . import LLMService

logger = getLogger(__name__)


class GeminiService(LLMService):
    def __init__(
        self,
        *,
        gemini_api_key: str = None,
        system_prompt: str = None,
        model: str = "gemini-2.0-flash-exp",
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
        genai.configure(api_key=gemini_api_key)
        generation_config = genai.GenerationConfig(
            temperature=temperature
        )
        self.gemini_client = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=system_prompt
        )
        self.contexts: List[Dict[str, List]] = {}

    def compose_messages(self, context_id: str, text: str) -> List[dict]:
        messages = []
        messages.extend(self.contexts.get(context_id, []))
        messages.append({"role": "user", "parts": [{"text": text}]})
        return messages

    def update_context(self, context_id: str, request_text: str, response_text: str):
        messages = self.contexts.get(context_id, [])
        messages.append({"role": "user", "parts": [{"text": request_text}]})
        messages.append({"role": "model", "parts": [{"text": response_text}]})
        self.contexts[context_id] = messages

    async def preflight(self):
        # Dummy request to initialize client (The first message takes long time)
        stream_resp = await self.gemini_client.generate_content_async("say just \"hello\"", stream=True)
        async for chunk in stream_resp:
            pass
        logger.info("Gemini client initialized.")

    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[str, None]:
        stream_resp = await self.gemini_client.generate_content_async(
            contents=messages,
            stream=True
        )

        async for chunk in stream_resp:
            if content := chunk.candidates[0].content.parts[0].text:
                yield content
