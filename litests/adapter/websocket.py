from abc import abstractmethod
import logging
from typing import Dict
from fastapi import APIRouter, WebSocket
from ..pipeline import LiteSTS
from .base import Adapter

logger = logging.getLogger(__name__)


class WebSocketSessionData:
    def __init__(self):
        self.id = None
        self.data = {}


class WebSocketAdapter(Adapter):
    def __init__(
        self,
        sts: LiteSTS = None
    ):
        super().__init__(sts)
        self.websockets: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, WebSocketSessionData] = {}

    @abstractmethod
    async def process_websocket(self, websocket: WebSocket, session_data: WebSocketSessionData):
        pass

    def get_websocket_router(self, path: str = "/ws"):
        router = APIRouter()

        @router.websocket(path)
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            session_data = WebSocketSessionData()

            try:
                while True:
                    await self.process_websocket(websocket, session_data)

            except Exception as ex:
                error_message = str(ex)

                if "WebSocket is not connected" in error_message:
                    logger.info(f"WebSocket disconnected (1): context_id={session_data.id}")
                elif "<CloseCode.NO_STATUS_RCVD: 1005>" in error_message:
                    logger.info(f"WebSocket disconnected (2): context_id={session_data.id}")
                else:
                    raise

            finally:
                if session_data.id:
                    await self.sts.finalize(session_data.id)
                    if session_data.id in self.websockets:
                        del self.websockets[session_data.id]
                    if session_data.id in self.sessions:
                        del self.sessions[session_data.id]

        return router
