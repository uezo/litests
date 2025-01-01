import base64
import json
from typing import Dict
from fastapi import FastAPI, WebSocket
from litests import LiteSTS
from litests.response_handler import ResponseHandler

OPENAI_API_KEY = "YOUR_API_KEY"
GOOGLE_API_KEY = "YOUR_API_KEY"


# Response handler for sending back to websocket client
class WebSocketResponseHandler(ResponseHandler):
    def __init__(self):
        self.websockets: Dict[str, WebSocket] = {}

    def set_websocket(self, session_id: str, websocket: WebSocket):
        self.websockets[session_id] = websocket

    async def handle_response(self, response):
        if response.type == "chunk" and response.audio_data:
            b64_chunk = base64.b64encode(response.audio_data).decode("utf-8")
            await self.websockets[response.context_id].send_json({
                "type": "chunk",
                "session_id": response.context_id,
                "text": response.text,
                "audio_data": b64_chunk
            })

    async def stop_response(self, context_id):
        if context_id in self.websockets:
            await self.websockets[context_id].send_json({
                "type": "stop",
                "session_id": context_id,
            })


# Create LiteSTS instance with response handler
sts = LiteSTS(
    vad_volume_db_threshold=-30,    # Adjust microphone sensitivity (Gate)
    stt_google_api_key=GOOGLE_API_KEY,
    llm_openai_api_key=OPENAI_API_KEY,
    response_handler=WebSocketResponseHandler(),
    debug=True
)


# Create websocket server with request handler
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        message_str = await websocket.receive_text()
        message = json.loads(message_str)
        message_type = message.get("type")
        session_id = message["session_id"]

        if message_type == "start":
            print(f"Connected: {session_id}")
            sts.response_handler.set_websocket(session_id, websocket)

        elif message_type == "data":
            b64_audio_data = message["audio_data"]
            audio_data = base64.b64decode(b64_audio_data)
            await sts.process_audio_samples(audio_data, session_id)

        elif message_type == "stop":
            print("stop")
            await websocket.close()
            break
