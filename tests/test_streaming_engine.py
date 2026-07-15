import queue
import threading

import numpy as np

from asr.events import Word
from asr.streaming_engine import StreamingASREngine
from audio.buffer import GrowingAudioBuffer

SR = 16000


class ScriptedTranscribe:
    """Returns pre-scripted hypotheses (lists of Words with LOCAL times)."""

    def __init__(self, hypotheses):
        self.hypotheses = list(hypotheses)
        self.calls = 0

    def __call__(self, audio):
        self.calls += 1
        if not self.hypotheses:
            return []
        if len(self.hypotheses) == 1:
            return list(self.hypotheses[0])
        return list(self.hypotheses.pop(0))


def _words(*texts, step=0.5, start=0.0):
    out = []
    t = start
    for text in texts:
        out.append(Word(text=text, start=t, end=t + step))
        t += step
    return out


def _make_engine(buf, transcribe, events, **overrides):
    params = dict(
        audio_buffer=buf,
        transcribe_fn=transcribe,
        events=events,
        stop_event=threading.Event(),
        chunk_interval_sec=1.0,
        min_audio_sec=1.0,
        commit_safety_margin_sec=0.1,
        max_buffer_sec=25.0,
        force_commit_keep_tail_words=2,
        silence_rms_threshold=0.0,  # disable the RMS gate unless a test enables it
        empty_decodes_before_trim=3,
        silence_keep_tail_sec=2.0,
    )
    params.update(overrides)
    return StreamingASREngine(**params)


def _noise(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * SR)) * 0.1).astype(np.float32)


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_agreement_commits_common_prefix_and_trims_buffer():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(_noise(3.0))
    events = queue.Queue()
    transcribe = ScriptedTranscribe([
        _words("hello", "brave", "world"),
        _words("hello", "brave", "cat"),
    ])
    engine = _make_engine(buf, transcribe, events)

    engine._iterate()  # first decode: no previous hypothesis, partial only
    first = _drain(events)
    assert [e.type for e in first] == ["partial"]

    duration_before = buf.duration_sec()
    engine._iterate()  # agreement on "hello brave"
    evs = _drain(events)
    commits = [e for e in evs if e.type == "commit"]
    partials = [e for e in evs if e.type == "partial"]
    assert len(commits) == 1
    assert commits[0].text == "hello brave"
    assert len(partials) == 1 and partials[0].text == "cat"
    assert buf.duration_sec() < duration_before  # trimmed up to the commit


def test_case_and_punctuation_do_not_break_agreement():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(_noise(3.0))
    events = queue.Queue()
    transcribe = ScriptedTranscribe([
        _words("Hello,", "world"),
        _words("hello", "World!"),
    ])
    engine = _make_engine(buf, transcribe, events)
    engine._iterate()
    engine._iterate()
    commits = [e for e in _drain(events) if e.type == "commit"]
    assert len(commits) == 1
    assert commits[0].text == "hello World!"  # latest decode's rendering wins


def test_no_agreement_emits_partial_only():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(_noise(3.0))
    events = queue.Queue()
    transcribe = ScriptedTranscribe([
        _words("alpha", "beta"),
        _words("gamma", "delta"),
    ])
    engine = _make_engine(buf, transcribe, events)
    engine._iterate()
    engine._iterate()
    evs = _drain(events)
    assert [e.type for e in evs] == ["partial", "partial"]


def test_force_commit_when_buffer_exceeds_max():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(_noise(4.0))
    events = queue.Queue()
    transcribe = ScriptedTranscribe([
        _words("a1", "a2", "a3", "a4", "a5"),
    ])
    engine = _make_engine(buf, transcribe, events, max_buffer_sec=3.0,
                          force_commit_keep_tail_words=2)
    # No previous hypothesis => no agreement, and the buffer already exceeds
    # max_buffer_sec => force-commit everything but the last 2 words.
    duration_before = buf.duration_sec()
    engine._iterate()
    evs = _drain(events)
    commits = [e for e in evs if e.type == "commit"]
    assert len(commits) == 1
    assert commits[0].text == "a1 a2 a3"
    assert buf.duration_sec() < duration_before  # trimmed aggressively


def test_silence_rms_gate_skips_transcribe_and_trims_after_n():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(np.zeros(6 * SR, dtype=np.float32))
    events = queue.Queue()
    transcribe = ScriptedTranscribe([_words("never")])
    engine = _make_engine(buf, transcribe, events,
                          silence_rms_threshold=1e-4,
                          empty_decodes_before_trim=3,
                          silence_keep_tail_sec=2.0)
    engine._iterate()
    engine._iterate()
    assert transcribe.calls == 0
    assert buf.duration_sec() == 6.0  # not yet trimmed
    engine._iterate()  # third empty decode triggers the trim
    assert transcribe.calls == 0
    assert buf.duration_sec() == 2.0  # only the tail kept


def test_empty_transcription_counts_towards_silence_trim():
    buf = GrowingAudioBuffer(sample_rate=SR)
    buf.write(_noise(6.0))
    events = queue.Queue()
    transcribe = ScriptedTranscribe([[]])  # audible but VAD/model yields nothing
    engine = _make_engine(buf, transcribe, events,
                          empty_decodes_before_trim=2,
                          silence_keep_tail_sec=2.0)
    engine._iterate()
    assert buf.duration_sec() == 6.0
    engine._iterate()
    assert transcribe.calls == 2
    assert buf.duration_sec() == 2.0
