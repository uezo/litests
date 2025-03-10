import asyncio
import struct
import pytest
from pathlib import Path

from litests.vad.standard import StandardSpeechDetector


@pytest.fixture
def test_output_dir(tmp_path: Path):
    """
    Temporary directory to store the file that is created in the each test case
    """
    return tmp_path


@pytest.fixture
def detector(test_output_dir):
    detector = StandardSpeechDetector(
        volume_db_threshold=-40.0,
        silence_duration_threshold=0.5,
        max_duration=3.0,
        min_duration=0.5,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        debug=True
    )

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, recorded_duration: float, session_id: str, session_data: dict):
        output_file = test_output_dir / f"speech_{session_id}.pcm"
        with open(output_file, "wb") as f:
            f.write(recorded_data)

    return detector


def generate_samples(amplitude: int, num_samples: int, sample_rate: int = 16000) -> bytes:
    data = [amplitude] * num_samples
    return struct.pack("<" + "h" * num_samples, *data)


@pytest.mark.asyncio
async def test_process_samples_speech_detection(detector, test_output_dir):
    """
    Test to verify that when data exceeding the volume threshold is provided, 
    recording starts, and after silence, recording ends, and on_speech_detected is called.
    """
    session_id = "test_session"

    # on_speech_detected will be invoked by loud samples longer than min_duration
    assert detector.min_duration == 0.5

    # Start with loud samples (0.5 sec, same as min_duration)
    loud_samples2 = generate_samples(amplitude=1200, num_samples=8000)
    await detector.process_samples(loud_samples2, session_id=session_id)
    session = detector.get_session(session_id)
    assert session.is_recording is True

    # Stop with silent samples
    silent_samples = generate_samples(amplitude=0, num_samples=16000)
    await detector.process_samples(silent_samples, session_id=session_id)

    # Wait for on_speech_detected invoked
    await asyncio.sleep(0.2)

    # Check whether the file that is created on_speech_detected exists
    output_file = test_output_dir / f"speech_{session_id}.pcm"
    assert output_file.exists(), "Recorded file doesn't exist"
    file_size = output_file.stat().st_size
    assert file_size > 0, "No data in the recorded file"


@pytest.mark.asyncio
async def test_process_samples_short_recording(detector, test_output_dir):
    """
    Verify that if recording starts but falls silent before min_duration, 
    on_speech_detected is not called, and no file is created.
    """
    session_id = "test_short"

    # on_speech_detected will be invoked by loud samples longer than min_duration
    assert detector.min_duration == 0.5

    # Loud samples slightly shorter than 0.5 (0.5 = 8000)
    loud_samples = generate_samples(amplitude=1000, num_samples=7999)
    await detector.process_samples(loud_samples, session_id=session_id)

    # Stop with silent samples
    silent_samples = generate_samples(amplitude=0, num_samples=8000)
    await detector.process_samples(silent_samples, session_id=session_id)

    await asyncio.sleep(0.2)

    output_file = test_output_dir / f"speech_{session_id}.pcm"
    assert not output_file.exists(), "File exists even the samples are shorter than min_duration"


@pytest.mark.asyncio
async def test_process_samples_max_duration(detector, test_output_dir):
    """
    Verify that when sound continues beyond max_duration (3 seconds), 
    recording is automatically stopped, and on_speech_detected is not called. 
    In the default implementation, recording is reset() when max_duration is exceeded, 
    and data is discarded.
    """
    session_id = "test_max_duration"

    assert detector.max_duration == 3.0

    # Loud samples as long max_duration (3.0 = 48000)
    loud_samples_long = generate_samples(amplitude=2000, num_samples=16000)
    await detector.process_samples(loud_samples_long, session_id=session_id)
    session = detector.get_session(session_id)
    # Make it sure that recording is started
    assert session.is_recording is True

    more_loud_samples_long = generate_samples(amplitude=2000, num_samples=32000)
    await detector.process_samples(more_loud_samples_long, session_id=session_id)
    session = detector.get_session(session_id)
    # Make it sure that recording is stopped
    assert session.is_recording is False

    await asyncio.sleep(0.2)

    output_file = test_output_dir / f"speech_{session_id}.pcm"
    assert not output_file.exists(), "File exists even the samples is as long as max_duration"


@pytest.mark.asyncio
async def test_process_stream(detector, test_output_dir):
    """
    Test stream processing via process_stream.
    """
    session_id = "test_stream"

    assert detector.min_duration == 0.5

    async def async_audio_stream():
        # Start recording with loud samples
        yield generate_samples(amplitude=1500, num_samples=16000)
        await asyncio.sleep(0.1)
        # More loud samples
        yield generate_samples(amplitude=1500, num_samples=3200)
        # Stop with silent samples
        yield generate_samples(amplitude=0, num_samples=16000)
        return

    await detector.process_stream(async_audio_stream(), session_id=session_id)

    # Wait for on_speech_detected invoked
    await asyncio.sleep(0.2)

    output_file = test_output_dir / f"speech_{session_id}.pcm"
    assert output_file.exists(), "Recorded file doesn't exist"
    file_size = output_file.stat().st_size
    assert file_size > 0, "No data in the recorded file"

    # Session is deleted after stream
    assert session_id not in detector.recording_sessions

@pytest.mark.asyncio
async def test_session_reset_and_delete(detector, test_output_dir):
    """
    Test the operation of reset / delete for a session.
    Reset clears buffers, etc. Delete removes the session.
    """
    session_id = "test_session_reset"

    assert detector.min_duration == 0.5
    loud_samples = generate_samples(amplitude=1000, num_samples=8000)

    # Start recording with loud samples
    await detector.process_samples(loud_samples, session_id=session_id)
    session = detector.get_session(session_id)
    assert session.is_recording is True
    assert len(session.buffer) > 0

    # Reset recording session
    detector.reset_session(session_id)
    session = detector.get_session(session_id)
    assert session.is_recording is False
    assert len(session.buffer) == 0

    # Start again
    await detector.process_samples(loud_samples, session_id=session_id)
    assert session.is_recording is True
    assert len(session.buffer) > 0

    # Delete session
    session = detector.get_session(session_id)
    detector.delete_session(session_id)
    session.is_recording = False
    assert len(session.buffer) == 0
    assert session_id not in detector.recording_sessions


@pytest.mark.asyncio
async def test_volume_threshold_change(detector, test_output_dir):
    """
    Verify that when volume_db_threshold is changed, amplitude_threshold is recalculated correctly. 
    Test whether providing actual audio affects the start of recording.
    """
    session_id = "test_threshold_change"

    detector.volume_db_threshold = -30.0
    new_amp_threshold = 32767 * (10 ** (-30.0 / 20.0))  # 1036.183520907373
    assert abs(detector.amplitude_threshold - new_amp_threshold) < 1.0

    # Under the threshold
    samples = generate_samples(amplitude=1000, num_samples=3200)
    await detector.process_samples(samples, session_id=session_id)

    session = detector.get_session(session_id)
    assert session.is_recording is False

    # Over the threshold
    samples = generate_samples(amplitude=1100, num_samples=3200)
    await detector.process_samples(samples, session_id=session_id)

    session = detector.get_session(session_id)
    assert session.is_recording is True


def test_session_data(detector):
    session_id_1 = "session_id_1"
    session_id_2 = "session_id_2"

    detector.set_session_data(session_id_1, "key", "val")
    assert detector.recording_sessions.get(session_id_1) is None

    detector.set_session_data(session_id_1, "key1", "val1", create_session=True)
    assert detector.recording_sessions.get(session_id_1).data == {"key1": "val1"}
    detector.set_session_data(session_id_1, "key2", "val2")
    assert detector.recording_sessions.get(session_id_1).data == {"key1": "val1", "key2": "val2"}

    assert detector.recording_sessions.get(session_id_2) is None
