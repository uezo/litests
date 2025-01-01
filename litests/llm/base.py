from abc import ABC, abstractmethod
import asyncio
import logging
import re
from typing import AsyncGenerator, List

logger = logging.getLogger(__name__)


class LLMResponse:
    def __init__(self, text: str = None, voice_text: str = None):
        self.text = text
        self.voice_text = voice_text


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
        skip_before: str = None
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
        self.skip_voice_before = skip_before

    def replace_last_option_split_char(self, original):
        return re.sub(self.option_split_chars_regex, r"\1|", original)

    @abstractmethod
    def compose_messages(self, context_id: str, text: str) -> List[dict]:
        pass

    @abstractmethod
    def update_context(self, context_id: str, request_text: str, response_text: str):
        pass

    @abstractmethod
    async def get_llm_stream_response(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        pass

    def to_voice_text(self, text: str) -> str:
        clean_text = text
        if self.skip_voice_before and self.skip_voice_before in clean_text:
            clean_text = text.split(self.skip_voice_before, 1)[1].strip()
        clean_text = re.sub(r"\[(\w+):([^\]]+)\]", "", clean_text)
        clean_text = re.sub(r"<(\w+)>|</(\w+)>", "", clean_text)
        clean_text = clean_text.strip()
        return clean_text

    async def chat_stream(self, context_id: str, text: str) -> AsyncGenerator[LLMResponse, None]:
        logger.info(f"User: {text}")
        messages = self.compose_messages(context_id, text)

        stream_buffer = ""
        response_text = ""
        skip_voice = True if self.skip_voice_before else False
        async for chunk in self.get_llm_stream_response(messages):
            stream_buffer += chunk

            for spc in self.split_chars:
                stream_buffer = stream_buffer.replace(spc, spc + "|")

            if len(stream_buffer) > self.option_split_threshold:
                stream_buffer = self.replace_last_option_split_char(stream_buffer)

            sp = stream_buffer.split("|")
            if len(sp) > 1: # >1 means `|` is found (splited at the end of sentence)
                sentence = sp.pop(0)
                stream_buffer = "".join(sp)
                if skip_voice:
                    if self.skip_voice_before in sentence:
                        skip_voice = False
                yield LLMResponse(sentence, None if skip_voice else self.to_voice_text(sentence))
                response_text += sentence

            await asyncio.sleep(0.001)   # wait slightly in every loop not to use up CPU

        if stream_buffer:
            yield LLMResponse(stream_buffer, None if skip_voice else self.to_voice_text(stream_buffer))
            response_text += stream_buffer

        logger.info(f"AI: {response_text}")
        self.update_context(context_id, text, response_text)
