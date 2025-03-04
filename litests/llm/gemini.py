import base64
from logging import getLogger
from typing import AsyncGenerator, Dict, List
import google.generativeai as genai
import httpx
from . import LLMService, LLMResponse, ToolCall
from .context_manager import ContextManager

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
        voice_text_tag: str = None,
        context_manager: ContextManager = None,
        debug: bool = False
    ):
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            split_chars=split_chars,
            option_split_chars=option_split_chars,
            option_split_threshold=option_split_threshold,
            voice_text_tag=voice_text_tag,
            context_manager=context_manager,
            debug=debug
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

    async def download_image(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    async def compose_messages(self, context_id: str, text: str, files: List[Dict[str, str]] = None) -> List[Dict]:
        messages = []

        # Extract the history starting from the first message where the role is 'user'
        histories = await self.context_manager.get_histories(context_id)
        while histories and histories[0]["role"] != "user":
            histories.pop(0)
        messages.extend(histories)

        parts = []
        if files:
            for f in files:
                if url := f.get("url"):
                    image_bytes = await self.download_image(url)
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    parts.append({"mime_type": "image/png", "data": image_b64})
        if text:
            parts.append({"text": text})

        messages.append({"role": "user", "parts": parts})
        return messages

    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        messages.append({"role": "model", "parts": [{"text": response_text}]})
        await self.context_manager.add_histories(context_id, messages, "gemini")

    async def preflight(self):
        # Dummy request to initialize client (The first message takes long time)
        stream_resp = await self.gemini_client.generate_content_async("say just \"hello\"", stream=True)
        async for chunk in stream_resp:
            pass
        logger.info("Gemini client initialized.")

    def register_tool(self, tool_function: callable):
        self.tools.append(tool_function)
        self.tool_functions[tool_function.__name__] = tool_function

    def tool(self, func):
        self.tools.append(func)
        self.tool_functions[func.__name__] = func
        return func

    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[LLMResponse, None]:
        stream_resp = await self.gemini_client.generate_content_async(
            contents=messages,
            tools=self.tools if self.tools else None,
            stream=True
        )

        tool_calls: List[ToolCall] = []
        async for chunk in stream_resp:
            for part in chunk.candidates[0].content.parts:
                if content := part.text:
                    yield LLMResponse(context_id=context_id, text=content)
                elif part.function_call:
                    tool_calls.append(ToolCall("", part.function_call.name, dict(part.function_call.args)))

        if tool_calls:
            # Do something before tool calls (e.g. say to user that it will take a long time)
            await self._on_before_tool_calls(tool_calls)

            # NOTE: Gemini 2.0 Flash doesn't return multiple tools at once for now (2025-01-07), but it's not explicitly documented.
            #       Multiple tools will be called sequentially: user -(llm)-> function_call -> function_response -(llm)-> function_call -> function_response -(llm)-> assistant
            # Execute tools
            for tc in tool_calls:
                yield LLMResponse(context_id=context_id, tool_call=tc)

                tool_result = await self.tool_functions[tc.name](**(tc.arguments))

                messages.append({
                    "role": "model",
                    "parts": [{
                        "function_call": {
                            "name": tc.name,
                            "args": tc.arguments
                        }
                    }]
                })

                messages.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": tc.name,
                            "response": tool_result
                        }
                    }]
                })

            async for llm_response in self.get_llm_stream_response(context_id, messages):
                yield llm_response
