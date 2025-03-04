import json
from logging import getLogger
from typing import AsyncGenerator, Dict, List
from litellm import acompletion
from . import LLMService, LLMResponse, ToolCall
from .context_manager import ContextManager

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
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt_by_user_prompt = system_prompt_by_user_prompt

    async def compose_messages(self, context_id: str, text: str, files: List[Dict[str, str]] = None) -> List[Dict]:
        messages = []
        if self.system_prompt:
            if self.system_prompt_by_user_prompt:
                messages.append({"role": "user", "content": self.system_prompt})
                messages.append({"role": "assistant", "content": "ok"})
            else:
                messages.append({"role": "system", "content": self.system_prompt})

        # Extract the history starting from the first message where the role is 'user'
        histories = await self.context_manager.get_histories(context_id)
        while histories and histories[0]["role"] != "user":
            histories.pop(0)
        messages.extend(histories)

        if files:
            content = []
            for f in files:
                if url := f.get("url"):
                    content.append({"type": "image_url", "image_url": {"url": url}})
            if text:
                content.append({"type": "text", "text": text})
        else:
            content = text
        messages.append({"role": "user", "content": content})

        return messages

    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        messages.append({"role": "assistant", "content": response_text})
        await self.context_manager.add_histories(context_id, messages, "chatgpt")

    def tool(self, spec: Dict):
        def decorator(func):
            tool_name = spec["function"]["name"]
            self.tools.append(spec)
            self.tool_functions[tool_name] = func
            return func
        return decorator

    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[LLMResponse, None]:
        stream_resp = await acompletion(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            tools=self.tools or None,
            stream=True
        )

        tool_calls: List[ToolCall] = []
        async for chunk in stream_resp:
            if content := chunk.choices[0].delta.content:
                yield LLMResponse(context_id=context_id, text=content)

            elif chunk.choices[0].delta.tool_calls:
                t = chunk.choices[0].delta.tool_calls[0]
                if t.id:
                    tool_calls.append(ToolCall(t.id, t.function.name, ""))
                if t.function.arguments:
                    tool_calls[-1].arguments += t.function.arguments

        if tool_calls:
            # Do something before tool calls (e.g. say to user that it will take a long time)
            await self._on_before_tool_calls(tool_calls)

            # Execute tools
            for tc in tool_calls:
                yield LLMResponse(context_id=context_id, tool_call=tc)

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

                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result),
                    "tool_call_id": tc.id
                })

            async for llm_response in self.get_llm_stream_response(context_id, messages):
                yield llm_response
