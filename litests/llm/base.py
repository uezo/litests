from abc import ABC, abstractmethod
import asyncio
import logging
import re
from typing import AsyncGenerator, List, Dict, Optional
from .context_manager import ContextManager, SQLiteContextManager

logger = logging.getLogger(__name__)


class ToolCall:
    def __init__(self, id: str = None, name: str = None, arguments: any = None):
        self.id = id
        self.name = name
        self.arguments = arguments


class LLMResponse:
    def __init__(self, context_id: str, text: str = None, voice_text: str = None, tool_call: ToolCall = None):
        self.context_id = context_id
        self.text = text
        self.voice_text = voice_text
        self.tool_call = tool_call


class LLMService(ABC):
    def __init__(
        self,
        *,
        system_prompt: str,
        model: str,
        temperature: float = 0.5,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        voice_text_tag: str = None,
        context_manager: ContextManager = None,
        debug: bool = False
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.split_chars = split_chars or ["。", "？", "！", ". ", "?", "!"]
        self.option_split_chars = option_split_chars or ["、", ", "]
        self.option_split_threshold = option_split_threshold
        self.split_patterns = []
        for char in self.option_split_chars:
            if char.endswith(" "):
                self.split_patterns.append(f"{re.escape(char)}")
            else:
                self.split_patterns.append(f"{re.escape(char)}\s?")
        self.option_split_chars_regex = f"({'|'.join(self.split_patterns)})\s*(?!.*({'|'.join(self.split_patterns)}))"
        self._request_filter = self.request_filter_default
        self.voice_text_tag = voice_text_tag
        self.tools = []
        self.tool_functions = {}
        self._on_before_tool_calls = self.on_before_tool_calls_default
        self.context_manager = context_manager or SQLiteContextManager()
        self.debug = debug

    # Decorators
    def request_filter(self, func):
        self._request_filter = func
        return func

    def request_filter_default(self, text: str) -> str:
        return text

    def tool(self, spec):
        def decorator(func):
            return func
        return decorator

    def on_before_tool_calls(self, func):
        self._on_before_tool_calls = func
        return func

    async def on_before_tool_calls_default(self, tool_calls: List[ToolCall]):
        pass

    def replace_last_option_split_char(self, original):
        return re.sub(self.option_split_chars_regex, r"\1|", original)

    @abstractmethod
    async def compose_messages(self, context_id: str, text: str) -> List[Dict]:
        pass

    @abstractmethod
    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        pass

    @abstractmethod
    async def get_llm_stream_response(self, messages: List[dict]) -> AsyncGenerator[LLMResponse, None]:
        pass

    def remove_control_tags(self, text: str) -> str:
        clean_text = text
        clean_text = re.sub(r"\[(\w+):([^\]]+)\]", "", clean_text)
        clean_text = clean_text.strip()
        return clean_text

    async def chat_stream(self, context_id: str, text: str) -> AsyncGenerator[LLMResponse, None]:
        logger.info(f"User: {text}")
        text = self._request_filter(text)
        logger.info(f"User(Filtered): {text}")

        messages = await self.compose_messages(context_id, text)
        message_length_at_start = len(messages) - 1

        stream_buffer = ""
        response_text = ""
        
        in_voice_tag = False
        target_start = f"<{self.voice_text_tag}>"
        target_end = f"</{self.voice_text_tag}>"

        def to_voice_text(segment: str) -> Optional[str]:
            if not self.voice_text_tag:
                return self.remove_control_tags(segment)

            nonlocal in_voice_tag
            if target_start in segment and target_end in segment:
                in_voice_tag = False
                start_index = segment.find(target_start)
                end_index = segment.find(target_end)
                voice_segment = segment[start_index + len(target_start): end_index]
                return self.remove_control_tags(voice_segment)

            elif target_start in segment:
                in_voice_tag = True
                start_index = segment.find(target_start)
                voice_segment = segment[start_index + len(target_start):]
                return self.remove_control_tags(voice_segment)

            elif target_end in segment:
                if in_voice_tag:
                    in_voice_tag = False
                    end_index = segment.find(target_end)
                    voice_segment = segment[:end_index]
                    return self.remove_control_tags(voice_segment)

            elif in_voice_tag:
                return self.remove_control_tags(segment)

            return None

        async for chunk in self.get_llm_stream_response(context_id, messages):
            if chunk.tool_call:
                yield chunk
                continue

            stream_buffer += chunk.text

            for spc in self.split_chars:
                stream_buffer = stream_buffer.replace(spc, spc + "|")

            if len(stream_buffer) > self.option_split_threshold:
                stream_buffer = self.replace_last_option_split_char(stream_buffer)

            segments = stream_buffer.split("|")
            while len(segments) > 1:
                sentence = segments.pop(0)
                stream_buffer = "|".join(segments)
                voice_text = to_voice_text(sentence)
                yield LLMResponse(context_id, sentence, voice_text)
                response_text += sentence
                segments = stream_buffer.split("|")

            await asyncio.sleep(0.001)   # wait slightly in every loop not to use up CPU

        if stream_buffer:
            voice_text = to_voice_text(stream_buffer)
            yield LLMResponse(context_id, stream_buffer, voice_text)
            response_text += stream_buffer

        logger.info(f"AI: {response_text}")
        if len(messages) > message_length_at_start:
            await self.update_context(
                context_id,
                messages[message_length_at_start - len(messages):],
                response_text,
            )
