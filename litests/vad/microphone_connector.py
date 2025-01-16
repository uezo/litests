import asyncio
from logging import getLogger
from .. import LiteSTS
from ..models import STSRequest

logger = getLogger(__name__)


async def start_with_pyaudio(
    session_id: str,
    sts: LiteSTS,
    sample_rate: int = 16000,
    channels: int = 1,
    chunk_size: int = 512
):
    logger.info("LiteSTS start listening to microphone. (pyaudio)")

    try:
        import pyaudio
    except ModuleNotFoundError:
        logger.warning("PyAudio is not found. Start text-based conversation in console.")
        while True:
            user_input = input("User: ")
            if not user_input.strip():
                continue
            await sts.invoke(STSRequest(context_id=session_id, text=user_input))

    # Start microphone stream
    p = pyaudio.PyAudio()
    pyaudio_stream = p.open(
        rate=sample_rate,
        channels=channels,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=chunk_size
    )

    # Process data
    loop = asyncio.get_running_loop()

    while True:
        data = await loop.run_in_executor(None, pyaudio_stream.read, chunk_size)
        if not data:
            break
        await sts.vad.process_samples(data, session_id)
        await asyncio.sleep(0.0001)

    # Finalize
    sts.vad.delete_session(session_id)

    logger.info("LiteSTS finish listening. (pyaudio)")


async def start_with_sounddevice(
    session_id: str,
    sts: LiteSTS,
    sample_rate: int = 16000,
    channels: int = 1,
    chunk_size: int = 512
):
    logger.info("LiteSTS start listening to microphone. (sd)")

    try:
        import numpy
        import sounddevice
    except ModuleNotFoundError:
        logger.warning("PyAudio is not found. Start text-based conversation in console.")
        while True:
            user_input = input("User: ")
            if not user_input.strip():
                continue
            await sts.invoke(STSRequest(context_id=session_id, text=user_input))

    # Start microphone stream
    sd_stream = sounddevice.InputStream(
        samplerate=sample_rate,
        channels=channels,
        blocksize=chunk_size
    )
    sd_stream.start()

    # Process data
    loop = asyncio.get_running_loop()

    while True:
        data, overflowed = await loop.run_in_executor(None, sd_stream.read, chunk_size)

        if overflowed:
            logger.warning("Microphone buffer overflowed. (sounddevice)")

        if data.size == 0:
            break

        audio_bytes = numpy.int16(data * 32767).tobytes()
        await sts.vad.process_samples(audio_bytes, session_id)
        await asyncio.sleep(0.0001)

    # Finalize
    sts.vad.delete_session(session_id)

    logger.info("LiteSTS finish listening. (sounddevice)")
