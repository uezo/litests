import json
from logging import getLogger
from typing import AsyncGenerator, Awaitable, Callable, Dict, List
import openai
from . import LLMService

logger = getLogger(__name__)


class ToolCall:
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.arguments = ""


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
        self.tools = []
        self.tool_functions = {}
        self.on_before_tool_calls: Callable[[List[ToolCall]], Awaitable[None]] = None

    def compose_messages(self, context_id: str, text: str) -> List[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.contexts.get(context_id, []))
        messages.append({"role": "user", "content": text})
        return messages

    def update_context(self, context_id: str, request_text: str, response_text: str):
        messages = self.contexts.get(context_id, [])
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            messages.append({"role": "user", "content": request_text})
        messages.append({"role": "assistant", "content": response_text})
        self.contexts[context_id] = messages

    def register_tool(self, tool_spec: dict, tool_function: callable):
        tool_name = tool_spec["function"]["name"]
        self.tools.append(tool_spec)
        self.tool_functions[tool_name] = tool_function

    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[str, None]:
        stream_resp = await self.openai_client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            tools=self.tools or None,
            stream=True
        )

        tool_calls: List[ToolCall] = []
        async for chunk in stream_resp:
            if chunk.choices[0].delta.tool_calls:
                t = chunk.choices[0].delta.tool_calls[0]
                if t.id:
                    tool_calls.append(ToolCall(t.id, t.function.name))
                if t.function.arguments:
                    tool_calls[-1].arguments += t.function.arguments

            elif content := chunk.choices[0].delta.content:
                yield content

        if tool_calls:
            if context_id not in self.contexts:
                self.contexts[context_id] = []
            
            # Add user message to context
            if messages[-1]["role"] != "tool":
                self.contexts[context_id].append(messages[-1])

            # Do something before tool calls (e.g. say to user that it will take a long time)
            if self.on_before_tool_calls:
                await self.on_before_tool_calls(tool_calls)

            # Execute tools
            for tc in tool_calls:
                tool_result = await self.tool_functions[tc.name](**(json.loads(tc.arguments)))

                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                        }
                    }]
                })
                self.contexts[context_id].append(messages[-1])

                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result),
                    "tool_call_id": tc.id
                })
                self.contexts[context_id].append(messages[-1])

            async for chunk in self.get_llm_stream_response(context_id, messages):
                yield chunk
