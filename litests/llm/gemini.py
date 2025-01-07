from logging import getLogger
from typing import AsyncGenerator, Awaitable, Callable, Dict, List
import google.generativeai as genai
from . import LLMService

logger = getLogger(__name__)


class ToolCall:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments


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
        self.tools: List[genai.types.Tool] = []
        self.tool_functions = {}
        self.on_before_tool_calls: Callable[[List[ToolCall]], Awaitable[None]] = None

    def compose_messages(self, context_id: str, text: str) -> List[dict]:
        messages = []
        messages.extend(self.contexts.get(context_id, []))
        messages.append({"role": "user", "parts": [{"text": text}]})
        return messages

    def update_context(self, context_id: str, request_text: str, response_text: str):
        messages = self.contexts.get(context_id, [])
        if len(messages) == 0 or "function_response" not in messages[-1]["parts"][0]:
            messages.append({"role": "user", "parts": [{"text": request_text}]})
        messages.append({"role": "model", "parts": [{"text": response_text}]})
        self.contexts[context_id] = messages

    async def preflight(self):
        # Dummy request to initialize client (The first message takes long time)
        stream_resp = await self.gemini_client.generate_content_async("say just \"hello\"", stream=True)
        async for chunk in stream_resp:
            pass
        logger.info("Gemini client initialized.")

    def register_tool(self, tool_function: callable):
        self.tools.append(tool_function)
        self.tool_functions[tool_function.__name__] = tool_function

    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[str, None]:
        stream_resp = await self.gemini_client.generate_content_async(
            contents=messages,
            tools=self.tools if self.tools else None,
            stream=True
        )

        tool_calls: List[ToolCall] = []
        async for chunk in stream_resp:
            for part in chunk.candidates[0].content.parts:
                if content := part.text:
                    yield content
                elif part.function_call:
                    tool_calls.append(ToolCall(part.function_call.name, dict(part.function_call.args)))

        if tool_calls:
            if context_id not in self.contexts:
                self.contexts[context_id] = []

            # Add user message to context
            if "function_response" not in messages[-1]["parts"][0]:
                self.contexts[context_id].append(messages[-1])

            # Do something before tool calls (e.g. say to user that it will take a long time)
            if self.on_before_tool_calls:
                await self.on_before_tool_calls(tool_calls)

            # NOTE: Gemini 2.0 Flash doesn't return multiple tools at once for now (2025-01-07), but it's not explicitly documented.
            #       Multiple tools will be called sequentially: user -(llm)-> function_call -> function_response -(llm)-> function_call -> function_response -(llm)-> assistant
            # Execute tools
            for tc in tool_calls:
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
                self.contexts[context_id].append(messages[-1])

                messages.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": tc.name,
                            "response": tool_result
                        }
                    }]
                })
                self.contexts[context_id].append(messages[-1])

            async for chunk in self.get_llm_stream_response(context_id, messages):
                yield chunk
