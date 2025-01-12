from logging import getLogger
import json
from typing import AsyncGenerator, Dict, List, Optional, Callable
import httpx
from . import LLMService

logger = getLogger(__name__)


class DifyService(LLMService):
    def __init__(
        self,
        *,
        api_key: str = None,
        user: str = None,
        base_url: str = "http://127.0.0.1",
        is_agent_mode: bool = False,
        make_inputs: callable = None,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        request_filter: Optional[Callable[[str], str]] = None,
        skip_before: str = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0
    ):
        super().__init__(
            system_prompt=None,
            model=None,
            temperature=0.0,
            split_chars=split_chars,
            option_split_chars=option_split_chars,
            option_split_threshold=option_split_threshold,
            request_filter=request_filter,
            skip_before=skip_before
        )
        self.conversation_ids: Dict[str, str] = {}
        self.api_key = api_key
        self.user = user
        self.base_url = base_url
        self.is_agent_mode = is_agent_mode
        self.make_inputs = make_inputs
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )

    async def compose_messages(self, context_id: str, text: str) -> List[Dict]:
        if self.make_inputs:
            inputs = self.make_inputs(context_id, text)
        else:
            inputs = {}

        return [{
            "inputs": inputs,
            "query": text,
            "response_mode": "streaming",
            "user": self.user,
            "auto_generate_name": False,
            "conversation_id": self.conversation_ids.get(context_id, "")
        }]

    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        # Context is managed at Dify server
        pass


    async def get_llm_stream_response(self, context_id: str, messages: List[dict]) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        stream_resp = await self.http_client.post(
            self.base_url + "/chat-messages",
            headers=headers,
            json=messages[0]
        )
        stream_resp.raise_for_status()

        message_event_value = "agent_message" if self.is_agent_mode else "message"
        async for chunk in stream_resp.aiter_lines():
            if chunk.startswith("data:"):
                chunk_json = json.loads(chunk[5:])
                if chunk_json["event"] == message_event_value:
                    answer = chunk_json["answer"]
                    yield answer
                elif chunk_json["event"] == "message_end":
                    # Save conversation id instead of managing context locally
                    self.conversation_ids[context_id] = chunk_json["conversation_id"]
