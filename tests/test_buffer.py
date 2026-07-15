import numpy as np
import pytest

from audio.buffer import GrowingAudioBuffer

SR = 16000


def _seconds(buf: GrowingAudioBuffer) -> float:
    return buf.duration_sec()


def test_write_and_snapshot_returns_copy_and_offset():
    buf = GrowingAudioBuffer(sample_rate=SR)
    data = np.ones(SR, dtype=np.float32)
    buf.write(data)

    audio, offset = buf.snapshot()
    assert offset == 0.0
    assert len(audio) == SR
    audio[0] = 42.0  # mutate the copy
    audio2, _ = buf.snapshot()
    assert audio2[0] == 1.0  # original untouched


def test_trim_until_drops_head_and_advances_offset():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(np.arange(2 * SR, dtype=np.float32))

    buf.trim_until(1.0)
    audio, offset = buf.snapshot()
    assert offset == pytest.approx(1.0)
    assert len(audio) == SR
    assert audio[0] == float(SR)  # first surviving sample


def test_trim_before_head_is_noop():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(np.zeros(SR, dtype=np.float32))
    buf.trim_until(0.5)
    buf.trim_until(0.2)  # before current head
    _, offset = buf.snapshot()
    assert offset == pytest.approx(0.5)
    assert _seconds(buf) == pytest.approx(0.5)


def test_trim_past_end_empties_buffer():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(np.zeros(SR, dtype=np.float32))
    buf.trim_until(10.0)
    audio, offset = buf.snapshot()
    assert len(audio) == 0
    assert offset == pytest.approx(1.0)  # advanced by what existed


def test_max_seconds_cap_drops_oldest_and_advances_offset():
    buf = GrowingAudioBuffer(sample_rate=SR, max_seconds=1.0)
    buf.write(np.zeros(SR, dtype=np.float32))
    buf.write(np.ones(SR // 2, dtype=np.float32))

    audio, offset = buf.snapshot()
    assert len(audio) == SR  # capped at 1s
    assert offset == pytest.approx(0.5)
    assert audio[-1] == 1.0


def test_end_time_sec_is_offset_plus_duration():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(np.zeros(2 * SR, dtype=np.float32))
    buf.trim_until(0.5)
    assert buf.end_time_sec() == pytest.approx(2.0)
