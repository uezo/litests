import base64
import io
import json
from uuid import uuid4
import wave
import httpx
import pyaudio


class AudioPlayer:
    def __init__(self, chunk_size: int = 1024):
        self.p = pyaudio.PyAudio()
        self.play_stream = None
        self.chunk_size = chunk_size

    def play(self, content: bytes):
        with wave.open(io.BytesIO(content), "rb") as wf:
            if not self.play_stream:
                self.play_stream = self.p.open(
                    format=self.p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )

            data = wf.readframes(self.chunk_size)
            while True:
                data = wf.readframes(self.chunk_size)
                if not data:
                    break
                self.play_stream.write(data)


audio_player = AudioPlayer()
context_id = str(uuid4())

while True:
    user_input = input("user: ")
    if not user_input.strip():
        continue

    with httpx.stream(
        method="post",
        url="http://127.0.0.1:8000/chat",
        json={
            "type": "start",
            "context_id": context_id,
            "text": user_input
        },
        timeout=60
    ) as resp:
        resp.raise_for_status()

        for chunk in resp.iter_lines():
            if chunk.startswith("data:"):
                chunk_json = json.loads(chunk[5:].strip())
                if chunk_json["type"] == "chunk":
                    print(f"assistant: {chunk_json['text']}")
                    if chunk_json["encoded_audio"]:
                        audio_bytes = base64.b64decode(chunk_json["encoded_audio"])
                        audio_player.play(audio_bytes)
