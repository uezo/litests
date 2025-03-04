import base64
import json
import logging
from typing import List
from uuid import uuid4
from pydantic import BaseModel
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse   # pip install sse-starlette
from litests import LiteSTS
from litests.models import STSRequest, STSResponse
from litests.adapter import Adapter

logger = logging.getLogger(__name__)


class File(BaseModel):
    url: str


class ChatRequest(BaseModel):
    context_id: str = None
    text: str = None
    audio_data: bytes = None
    audio_duration: float = 0
    files: List[File] = None

    def to_sts_request(self) -> STSRequest:
        return STSRequest(
            context_id=self.context_id,
            text=self.text,
            audio_data=self.audio_data,
            audio_duration=self.audio_duration,
            files=[{"url": f.url} for f in self.files] if self.files else None
        )


class ChatChunkResponse(BaseModel):
    type: str
    context_id: str
    text: str|None = None
    voice_text: str|None = None
    tool_call: str|None = None
    encoded_audio: str|None = None


class HttpAdapter(Adapter):
    def __init__(self, sts: LiteSTS):
        super().__init__(sts)

    def get_api_router(self, path: str = "/chat"):
        router = APIRouter()

        @router.post(path)
        async def post_chat(request: ChatRequest):
            if not request.context_id:
                request.context_id = str(uuid4())

            async def stream_response():
                async for chunk in self.sts.invoke(request.to_sts_request()):
                    try:
                        if chunk.audio_data:
                            b64_audio= base64.b64encode(chunk.audio_data).decode("utf-8")
                        else:
                            b64_audio = None

                        yield ChatChunkResponse(
                            type=chunk.type,
                            context_id=chunk.context_id,
                            text=chunk.text,
                            voice_text=chunk.voice_text,
                            tool_call=json.dumps(chunk.tool_call.__dict__) if chunk.tool_call else None,
                            encoded_audio=b64_audio
                        ).model_dump_json()
                    
                    except Exception as ex:
                        logger.error(f"Error at HTTP adapter: {ex}")

            return EventSourceResponse(stream_response())

        return router

    async def handle_response(self, response: STSResponse):
        pass

    async def stop_response(self, context_id: str):
        pass
