from abc import ABC, abstractmethod
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
import re
import sqlite3
from typing import AsyncGenerator, List, Dict

logger = logging.getLogger(__name__)


import sqlite3
import json
import logging
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict

logger = logging.getLogger(__name__)


class ContextManager(ABC):
    @abstractmethod
    async def get_histories(self, context_id: str, limit: int = 100) -> List[Dict]:
        pass

    @abstractmethod
    async def add_histories(self, context_id: str, data_list: List[Dict], context_schema: str = None):
        pass


class SQLiteContextManager(ContextManager):
    def __init__(self, db_path="context.db", context_timeout=3600):
        self.db_path = db_path
        self.context_timeout = context_timeout
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                # Create table if not exists
                # (Primary key 'id' automatically gets indexed by SQLite)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_histories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TIMESTAMP NOT NULL,
                        context_id TEXT NOT NULL,
                        serialized_data JSON NOT NULL,
                        context_schema TEXT
                    )
                    """
                )

                # Create an index to speed up filtering queries by context_id and created_at
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_histories_context_id_created_at
                    ON chat_histories (context_id, created_at)
                    """
                )

        except Exception as ex:
            logger.error(f"Error at init_db: {ex}")
        finally:
            conn.close()

    async def get_histories(self, context_id: str, limit: int = 100) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            sql = """
            SELECT serialized_data
            FROM chat_histories
            WHERE context_id = ?
            """
            params = [context_id]

            if self.context_timeout > 0:
                # Cutoff time to exclude old records
                sql += " AND created_at >= ?"
                cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.context_timeout)
                params.append(cutoff_time)

            sql += " ORDER BY id DESC"

            if limit > 0:
                sql += " LIMIT ?"
                params.append(limit)

            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

            # Reverse the list so that the newest item is at the end (larger index)
            rows.reverse()
            results = [json.loads(row[0]) for row in rows]
            return results

        except Exception as ex:
            logger.error(f"Error at get_histories: {ex}")
            return []

        finally:
            conn.close()

    async def add_histories(self, context_id: str, data_list: List[Dict], context_schema: str = None):
        if not data_list:
            # If the list is empty, do nothing
            return

        conn = sqlite3.connect(self.db_path)
        try:
            # Prepare INSERT statement
            columns = ["created_at", "context_id", "serialized_data", "context_schema"]
            placeholders = ["?"] * len(columns)
            sql = f"""
                INSERT INTO chat_histories ({', '.join(columns)}) 
                VALUES ({', '.join(placeholders)})
            """

            now_utc = datetime.now(timezone.utc)
            records = []
            for data_item in data_list:
                record = (
                    now_utc,                        # created_at
                    context_id,                     # context_id
                    json.dumps(data_item, ensure_ascii=True),  # serialized_data
                    context_schema,                 # context_schema
                )
                records.append(record)

            # Execute many inserts in a single statement
            conn.executemany(sql, records)
            conn.commit()

        except Exception as ex:
            logger.error(f"Error at add_histories: {ex}")
            conn.rollback()

        finally:
            conn.close()


class ToolCall:
    def __init__(self, id: str = None, name: str = None, arguments: any = None):
        self.id = id
        self.name = name
        self.arguments = arguments


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
        skip_before: str = None,
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
        self.skip_voice_before = skip_before
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
        text = self._request_filter(text)
        logger.info(f"User(Filtered): {text}")

        messages = await self.compose_messages(context_id, text)
        message_length_at_start = len(messages) - 1

        stream_buffer = ""
        response_text = ""
        skip_voice = True if self.skip_voice_before else False
        async for chunk in self.get_llm_stream_response(context_id, messages):
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
            if skip_voice:
                if self.skip_voice_before in stream_buffer:
                    skip_voice = False
            yield LLMResponse(stream_buffer, None if skip_voice else self.to_voice_text(stream_buffer))
            response_text += stream_buffer

        logger.info(f"AI: {response_text}")
        if len(messages) > message_length_at_start:
            await self.update_context(context_id, messages[message_length_at_start - len(messages):], response_text)
