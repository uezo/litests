import json
from logging import getLogger
from typing import AsyncGenerator, Dict, List
from anthropic import AsyncAnthropic
from . import LLMService, LLMResponse, ToolCall
from .context_manager import ContextManager

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
        self.anthropic_client = AsyncAnthropic(
            api_key=anthropic_api_key,
            base_url=base_url
        )
        self.max_tokens = max_tokens

    async def compose_messages(self, context_id: str, text: str, files: List[Dict[str, str]] = None) -> List[Dict]:
        messages = []

        # Extract the history starting from the first message where the role is 'user'
        histories = await self.context_manager.get_histories(context_id)
        while histories and histories[0]["role"] != "user":
            histories.pop(0)
        messages.extend(histories)

        content = []
        if files:
            for f in files:
                if url := f.get("url"):
                    content.append({"type": "image", "source": {"type": "url", "url": url}})
        if text:
            content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content})

        return messages

    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
        await self.context_manager.add_histories(context_id, messages, "claude")

    def tool(self, spec: Dict):
        def decorator(func):
            self.tools.append(spec)
            self.tool_functions[spec["name"]] = func
            return func
        return decorator

    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[LLMResponse, None]:
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
                        tool_calls.append(ToolCall(chunk.content_block.id, chunk.content_block.name, ""))
                elif chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        response_text += chunk.delta.text
                        yield LLMResponse(context_id=context_id, text=chunk.delta.text)
                    elif chunk.delta.type == "input_json_delta":
                        tool_calls[-1].arguments += chunk.delta.partial_json

        if tool_calls:
            # Do something before tool calls (e.g. say to user that it will take a long time)
            await self._on_before_tool_calls(tool_calls)

            # NOTE: Claude 3.5 Sonnet doesn't return multiple tools at once for now (2025-01-07), but it's not explicitly documented.
            #       Multiple tools will be called sequentially: user -(llm)-> tool_use -> tool_result -(llm)-> tool_use -> tool_result -(llm)-> assistant
            # Execute tools
            for tc in tool_calls:
                yield LLMResponse(context_id=context_id, tool_call=tc)

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

            async for llm_response in self.get_llm_stream_response(context_id, messages):
                yield llm_response
