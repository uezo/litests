import asyncio
import base64
import io
import json
import queue
import threading
import uuid
import wave
import pyaudio
import websockets

WS_URL = "ws://localhost:8000/ws"
CANCEL_ECHO = True


class AudioPlayer:
    def __init__(self, chunk_size: int = 1024):
        self.queue = queue.Queue()

        self.thread = threading.Thread(target=self.process_queue, daemon=True)
        self.thread.start()

        self.to_wave = None
        self.p = pyaudio.PyAudio()
        self.play_stream = None
        self.chunk_size = chunk_size

        self.is_playing = False

    def play(self, content: bytes):
        try:
            self.is_playing = True

            if self.to_wave:
                wave_content = self.to_wave(content)
            else:
                wave_content = content

            with wave.open(io.BytesIO(wave_content), "rb") as wf:
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

        finally:
            self.is_playing = False

    def process_queue(self):
        while True:
            data = self.queue.get()
            if data is None:
                break

            self.play(data)

    def add(self, audio_bytes: bytes):
        self.queue.put(audio_bytes)

    def cancel(self):
        while not self.queue.empty():
            self.queue.get()

    def stop(self):
        self.queue.put(None)
        self.thread.join()
        if self.play_stream:
            self.play_stream.stop_stream()
            self.play_stream.close()
        self.p.terminate()

audio_player = AudioPlayer()


# Send microphone data to Speech-to-Speech server
async def send_microphone_data(ws, session_id: str, cancel_echo: bool):
    p = pyaudio.PyAudio()

    mic_stream = p.open(
        format=pyaudio.paInt16, # LINEAR16
        channels=1, # Mono
        rate=16000,
        input=True,
        frames_per_buffer=512
    )

    while True:
        data = mic_stream.read(512, exception_on_overflow=False)
        b64_data = base64.b64encode(data).decode("utf-8")

        if not (audio_player.is_playing and cancel_echo):
            await ws.send(json.dumps({
                "type": "data",
                "session_id": session_id,
                "audio_data": b64_data
            }))

        await asyncio.sleep(0.01)


# Receive and play audio from Speech-to-Speech server
async def receive_and_play_audio(ws):
    while True:
        message_str = await ws.recv()
        message = json.loads(message_str)
        message_type = message.get("type")

        if message_type == "chunk":
            print(f"Response: {message.get('text')}")
            b64_data = message.get("audio_data")
            if b64_data:
                audio_bytes = base64.b64decode(b64_data)
                audio_player.add(audio_bytes)

        elif message_type == "stop":
            print(f"Stop requested")
            audio_player.cancel()


async def main():
    async with websockets.connect(WS_URL) as ws:
        session_id = str(uuid.uuid4())

        # Send start message
        start_message = {
            "type": "start",
            "session_id": session_id
        }
        await ws.send(json.dumps(start_message))

        print(f"Connected: {session_id}")

        # Start send and receive task
        send_task = asyncio.create_task(send_microphone_data(ws, session_id, CANCEL_ECHO))
        receive_task = asyncio.create_task(receive_and_play_audio(ws))
        await asyncio.gather(send_task, receive_task)

        # Send stop message
        await ws.send(json.dumps({
            "type": "stop",
            "session_id": session_id
        }))


if __name__ == "__main__":
    asyncio.run(main())
