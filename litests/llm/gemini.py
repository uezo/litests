import base64
from logging import getLogger
from typing import AsyncGenerator, Dict, List
from google import genai
from google.genai import types
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
        model: str = "gemini-2.0-flash",
        temperature: float = 0.5,
        thinking_budget: int = -1,
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
        self.gemini_client = genai.Client(
            api_key=gemini_api_key
        )
        self.thinking_budget = thinking_budget

    async def download_image(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    async def compose_messages(self, context_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> List[Dict]:
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
                    parts.append(types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    ))
        if text:
            parts.append(types.Part.from_text(text=text))

        messages.append(types.Content(role="user", parts=parts))
        return messages

    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        messages.append(types.Content(role="model", parts=[types.Part.from_text(text=response_text)]))
        dict_messages = []
        for m in messages:
            dumped = m.model_dump()
            for part in dumped.get("parts", []):
                inline_data = part.get("inline_data")
                if inline_data and "data" in inline_data:
                    inline_data["data"] = base64.b64encode(inline_data["data"]).decode("utf-8")
            dict_messages.append(dumped)
        await self.context_manager.add_histories(context_id, dict_messages, "gemini")

    async def preflight(self):
        # Dummy request to initialize client (The first message takes long time)
        stream_resp = await self.gemini_client.aio.models.generate_content_stream(
            model=self.model,
            contents="say just \"hello\""
        )
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

    async def get_llm_stream_response(self, context_id: str, user_id: str, messages: List[dict], system_prompt_params: Dict[str, any] = None) -> AsyncGenerator[LLMResponse, None]:
        if self.thinking_budget >= 0:
            thinking_config = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        else:
            thinking_config = None

        stream_resp = await self.gemini_client.aio.models.generate_content_stream(
            model=self.model,
            config = types.GenerateContentConfig(
                system_instruction=self.get_system_prompt(system_prompt_params),
                temperature=self.temperature,
                tools=self.tools if self.tools else None,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=thinking_config
            ),
            contents=messages,
        )

        tool_calls: List[ToolCall] = []
        async for chunk in stream_resp:
            for part in chunk.candidates[0].content.parts:
                if content := part.text:
                    yield LLMResponse(context_id=context_id, text=content)
                elif part.function_call:
                    tool_calls.append(ToolCall(part.function_call.id, part.function_call.name, dict(part.function_call.args)))

        if tool_calls:
            # Do something before tool calls (e.g. say to user that it will take a long time)
            await self._on_before_tool_calls(tool_calls)

            # NOTE: Gemini 2.0 Flash doesn't return multiple tools at once for now (2025-01-07), but it's not explicitly documented.
            #       Multiple tools will be called sequentially: user -(llm)-> function_call -> function_response -(llm)-> function_call -> function_response -(llm)-> assistant
            # Execute tools
            for tc in tool_calls:
                yield LLMResponse(context_id=context_id, tool_call=tc)

                tool_result = await self.execute_tool(tc.name, tc.arguments, {"user_id": user_id})

                messages.append(types.Content(
                    role="model",
                    parts=[types.Part.from_function_call(name=tc.name, args=tc.arguments)]
                ))
                messages.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=tc.name, response=tool_result)]
                ))

            async for llm_response in self.get_llm_stream_response(context_id, user_id, messages):
                yield llm_response
