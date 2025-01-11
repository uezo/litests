import json
from logging import getLogger
from typing import AsyncGenerator, Awaitable, Callable, Dict, List
from anthropic import AsyncAnthropic
from . import LLMService, ContextManager


logger = getLogger(__name__)


class ToolCall:
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.arguments = ""


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
        skip_before: str = None,
        context_manager: ContextManager = None
    ):
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            split_chars=split_chars,
            option_split_chars=option_split_chars,
            option_split_threshold=option_split_threshold,
            skip_before=skip_before,
            context_manager=context_manager
        )
        self.anthropic_client = AsyncAnthropic(
            api_key=anthropic_api_key,
            base_url=base_url
        )
        self.max_tokens = max_tokens
        self.tools = []
        self.tool_functions = {}
        self.on_before_tool_calls: Callable[[List[ToolCall]], Awaitable[None]] = None

    async def compose_messages(self, context_id: str, text: str) -> List[Dict]:
        messages = []

        # Extract the history starting from the first message where the role is 'user'
        histories = await self.context_manager.get_histories(context_id)
        while histories and histories["role"] != "user":
            histories.pop(0)
        messages.extend(histories)

        messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        return messages

    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
        await self.context_manager.add_histories(context_id, messages, "claude")

    def register_tool(self, tool_spec: dict, tool_function: callable):
        self.tools.append(tool_spec)
        self.tool_functions[tool_spec["name"]] = tool_function

    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[str, None]:
        async with self.anthropic_client.messages.stream(
            messages=messages,
            system=self.system_prompt or "",
            model=self.model,
            temperature=self.temperature,
            tools=self.tools,
            max_tokens=self.max_tokens
        ) as stream_resp:
            tool_calls: List[ToolCall] = []
            response_text = ""
            async for chunk in stream_resp:
                if chunk.type == "content_block_start":
                    if chunk.content_block.type == "tool_use":
                        tool_calls.append(ToolCall(chunk.content_block.id, chunk.content_block.name))
                elif chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        response_text += chunk.delta.text
                        yield chunk.delta.text
                    elif chunk.delta.type == "input_json_delta":
                        tool_calls[-1].arguments += chunk.delta.partial_json

        if tool_calls:
            # Do something before tool calls (e.g. say to user that it will take a long time)
            if self.on_before_tool_calls:
                await self.on_before_tool_calls(tool_calls)

            # NOTE: Claude 3.5 Sonnet doesn't return multiple tools at once for now (2025-01-07), but it's not explicitly documented.
            #       Multiple tools will be called sequentially: user -(llm)-> tool_use -> tool_result -(llm)-> tool_use -> tool_result -(llm)-> assistant
            # Execute tools
            for tc in tool_calls:
                arguments_json = json.loads(tc.arguments)
                tool_result = await self.tool_functions[tc.name](**arguments_json)

                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": arguments_json
                    }]
                })

                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps(tool_result)
                    }]
                })

            async for chunk in self.get_llm_stream_response(context_id, messages):
                yield chunk
