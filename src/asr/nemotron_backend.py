from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional

import numpy as np

from app_config import NemotronConfig
from asr.events import ASREvent, Word

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# A word is closed when the next one starts. Without a second rule the last
# word before a pause would sit in the assembler until someone speaks again,
# so this many seconds' worth of blank frames also closes it. Blanks arrive
# once per encoder frame while audio flows, so this is stream time, not wall
# time. It has to be generous: sentence-final punctuation is emitted only
# after the model has heard the pause, ~1.5 s late in practice, and closing
# the word earlier turns "встреча?" into two words.
IDLE_FLUSH_SEC = 2.0

# The rule above needs frames, and WASAPI loopback delivers none once nothing
# is rendering audio - the pending word would hang until playback resumes.
# This wall-clock fallback runs while we wait on an empty frame queue.
STALE_FLUSH_SEC = 1.5


def _has_content(text: str) -> bool:
    """True for anything that can stand as a word on its own."""
    return any(ch.isalnum() for ch in text)


@dataclass(slots=True)
class _Pending:
    text: str
    start_frame: int
    end_frame: int


class WordAssembler:
    """
    Groups streamed RNN-T token pieces into whole words.

    The tokenizer is subword: one word arrives as several pieces, and the
    *leading space* is the only marker of a word boundary ("` архитект`",
    "`уру`"). A lone `" "` piece is a boundary too. Dropping "empty" pieces
    glues words together ("обсудимархитектурунового") - that is the first
    thing a naive implementation gets wrong. Punctuation has no leading space,
    so it sticks to the word before it, which is what we want.

    Frames are subsampled encoder frames (80 ms each); the caller passes their
    index, we turn it into global stream seconds. `end` is the emission point
    of the last piece + one frame, NOT the acoustic end of the word - RNN-T
    emits a token at a frame, not over a span.
    """

    def __init__(
        self,
        frame_sec: float,
        idle_flush_sec: float = IDLE_FLUSH_SEC,
        on_late_punct: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self._frame_sec = float(frame_sec)
        self._idle_frames = max(1, int(round(idle_flush_sec / self._frame_sec)))
        self._pending: Optional[_Pending] = None
        # Called with (punctuation, emission_time_sec) for a mark that arrived
        # after its word was already committed (this model emits sentence-final
        # '.'/'?' ~1.5 s late, after the pause an idle/stale flush has already
        # acted on). The time is the mark's own emission frame in global stream
        # seconds; the backend turns it into an `amend` event that binds by
        # timestamp instead of dropping the mark or pinning it to the last word.
        self._on_late_punct = on_late_punct

    def push(self, piece: str, frame: int) -> List[Word]:
        """Feed one decoded token piece. Returns the words it completed."""
        if not piece:
            return []

        done: List[Word] = []
        opens_word = piece.startswith(" ")
        text = piece.strip()

        if opens_word and self._pending is not None:
            done.append(self._finish())
        if not text:
            # Lone separator: it carries no letters, it only says that the
            # next piece starts a fresh word.
            self._pending = None
            return done
        if self._pending is None and not opens_word and not _has_content(text):
            # Punctuation for a word that an idle/stale flush already committed
            # (the model emits it only after hearing the pause). The word is
            # gone from here, so hand the punctuation to the backend to append
            # onto the last committed word instead of dropping it.
            if self._on_late_punct is not None:
                self._on_late_punct(text, frame * self._frame_sec)
            return done
        if opens_word or self._pending is None:
            self._pending = _Pending(text=text, start_frame=frame, end_frame=frame)
        else:
            self._pending.text += text
            self._pending.end_frame = frame
        return done

    def advance(self, frame: int) -> List[Word]:
        """Called on blank emissions. Closes a word that has gone quiet."""
        p = self._pending
        if p is not None and frame - p.end_frame >= self._idle_frames:
            return [self._finish()]
        return []

    def flush(self) -> List[Word]:
        """Close whatever is pending (end of stream)."""
        return [self._finish()] if self._pending is not None else []

    def _finish(self) -> Word:
        p = self._pending
        self._pending = None
        return Word(
            text=p.text,
            start=p.start_frame * self._frame_sec,
            end=(p.end_frame + 1) * self._frame_sec,
        )


class NemotronBackend:
    """
    Cache-aware streaming backend (nvidia/nemotron-3.5-asr-streaming-0.6b).

    Push-style: the capture supervisor hands us resampled 16 kHz mono chunks
    through `frame_sink`, we cut them into the fixed-size mel chunks the model
    demands and feed a *single* long-lived `model.generate()` call, which pulls
    from our generator and blocks when we have no audio.

    The output is append-only - the model never rewrites what it already
    emitted - so this path has no `partial` events, no LocalAgreement-2 and no
    GrowingAudioBuffer. It also needs no external punctuation: casing and
    punctuation come from the model.
    """

    name = "nemotron"

    def __init__(
        self,
        *,
        cfg: NemotronConfig,
        frame_sink: "queue.Queue[np.ndarray]",
        latency=None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cfg = cfg
        self._frames = frame_sink
        self._latency = latency
        self._on_status = on_status

        self._model = None
        self._processor = None
        self._prompt_ids = None
        self._blank_ids: set[int] = set()
        self._frame_sec = 0.08
        self._first_chunk_samples = 0
        self._chunk_samples = 0

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        """Instantiate model + processor. Runs on the asr-loader thread."""
        cfg = self._cfg
        self._status(f"Загружаю Nemotron ({cfg.model}, {cfg.device}/{cfg.dtype})…")

        # Heavy imports stay here: importing torch/transformers costs seconds
        # and the whisper path must not pay for them.
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        processor = AutoProcessor.from_pretrained(cfg.model)
        processor.set_num_lookahead_tokens(cfg.lookahead_tokens)
        if cfg.language not in processor.prompt_dictionary:
            raise ValueError(
                f"asr.nemotron.language={cfg.language!r} is not supported by the model "
                f"(use 'auto', 'ru', 'en-US', ...)"
            )

        dtype = getattr(torch, cfg.dtype)
        model = AutoModelForRNNT.from_pretrained(cfg.model, dtype=dtype)
        model = model.to(cfg.device).eval()

        self._processor = processor
        self._model = model
        self._prompt_ids = torch.tensor(
            [processor.prompt_dictionary[cfg.language]], dtype=torch.long, device=model.device
        )
        # `decode()` skips these; the config's blank also drives frame advance.
        self._blank_ids = {
            int(model.config.blank_token_id),
            int(processor.blank_token_id),
            int(processor.tokenizer.pad_token_id),
        }
        self._frame_sec = processor._encoder_frame_ms / 1000.0

        hop = processor.feature_extractor.hop_length
        # NOTE: `processor.num_samples_first_audio_chunk` returns one mel frame
        # too many for the size `generate()` validates (49 required vs 50
        # produced at lookahead 6), so the first chunk is sized by hand.
        self._first_chunk_samples = max(1, (processor.num_mel_frames_first_audio_chunk - 1) * hop)
        self._chunk_samples = processor.num_samples_per_audio_chunk

        logger.info(
            "Nemotron loaded: %s on %s/%s, lookahead=%d (%d ms), language=%s",
            cfg.model, cfg.device, cfg.dtype, cfg.lookahead_tokens,
            processor.streaming_latency_ms, cfg.language,
        )

    # -- loop ------------------------------------------------------------

    def run(
        self,
        *,
        events: "queue.Queue[ASREvent]",
        stop_event: threading.Event,
    ) -> None:
        if self._model is None or self._processor is None:
            raise RuntimeError("NemotronBackend.run() called before load()")

        self._emit_log(events, "engine started (nemotron)")
        try:
            import torch

            assembler = WordAssembler(
                self._frame_sec,
                on_late_punct=lambda punct, time_sec: self._amend(events, punct, time_sec),
            )
            streamer = _WordStreamer(
                model=self._model,
                processor=self._processor,
                assembler=assembler,
                blank_ids=self._blank_ids,
                blank_id=int(self._model.config.blank_token_id),
                on_words=lambda words: self._commit(events, words),
            )
            chunks = self._mel_chunks(stop_event, streamer)
            try:
                first = next(chunks)
            except StopIteration:
                # Stopped before any audio arrived. `generate()` would raise on
                # an empty generator, and a clean exit is not a fatal error.
                return

            with torch.inference_mode():
                self._model.generate(
                    input_features=_chunks_from(first, chunks),
                    num_lookahead_tokens=self._cfg.lookahead_tokens,
                    prompt_ids=self._prompt_ids,
                    streamer=streamer,
                )
            self._commit(events, assembler.flush())
        except Exception as exc:  # noqa: BLE001 - the thread must report, not die
            logger.exception("nemotron engine failed")
            self._emit_fatal(events, f"nemotron engine failed: {exc!r}")
            stop_event.set()
        finally:
            self._emit_log(events, "engine stopped (nemotron)")

    def _mel_chunks(self, stop_event: threading.Event, streamer: "_WordStreamer") -> Iterator[Any]:
        """Yield fixed-size mel chunks, blocking on the frame queue.

        `generate()` pulls one chunk whenever the decoder has consumed the
        previous one, so blocking here is how a live stream stalls in silence.
        Chunk sizes are exact: cache-aware streaming rejects anything else,
        which is why the final chunk is zero-padded instead of dropped.
        """
        processor = self._processor
        pending = np.empty(0, dtype=np.float32)
        first = True

        while True:
            need = self._first_chunk_samples if first else self._chunk_samples
            while len(pending) < need and not stop_event.is_set():
                try:
                    chunk = self._frames.get(timeout=0.2)
                except queue.Empty:
                    # Waiting on an empty queue is the only moment we know the
                    # audio stream itself went quiet.
                    streamer.flush_if_stale(STALE_FLUSH_SEC)
                    continue
                pending = np.concatenate((pending, np.asarray(chunk, dtype=np.float32)))

            if stop_event.is_set():
                if len(pending) == 0:
                    return
                # Grab 5 of the model-card example: its loop drops the last
                # partial chunk and cuts the tail of the phrase.
                pending = np.concatenate(
                    (pending, np.zeros(need - len(pending), dtype=np.float32))
                )

            block, pending = pending[:need], pending[need:]
            features = processor(
                block,
                sampling_rate=SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=first,
                language=self._cfg.language,
            )
            first = False
            # Control comes back here only when the decoder has consumed the
            # chunk, so the gap measures the model's work on one chunk.
            t_yield = time.perf_counter()
            yield features["input_features"]
            if self._latency is not None:
                self._latency.mark("nemotron", time.perf_counter() - t_yield)

            if stop_event.is_set():
                return

    # -- helpers ---------------------------------------------------------

    def _commit(self, events: "queue.Queue[ASREvent]", words: List[Word]) -> None:
        if not words:
            return
        text = " ".join(w.text for w in words if w.text).strip()
        try:
            events.put(ASREvent(type="commit", source="engine", text=text, words=words), timeout=0.5)
        except queue.Full:
            self._emit_log(events, "event queue full on commit; dropped")

    def _amend(self, events: "queue.Queue[ASREvent]", punct: str, time_sec: float) -> None:
        """Append late punctuation to the committed word it belongs to.

        Carries the mark's own emission time (global stream seconds) so the
        splitter can bind it to the word whose end is closest, not blindly to
        the last one - by the time the mark arrives 1-2 later words may have
        committed. Emitted through the same queue as commits, so the target
        word is already in the store when this is applied.
        """
        punct = punct.strip()
        if not punct:
            return
        try:
            events.put(
                ASREvent(type="amend", source="engine", text=punct, time_sec=time_sec),
                timeout=0.5,
            )
        except queue.Full:
            self._emit_log(events, "event queue full on amend; dropped")

    def _status(self, msg: str) -> None:
        if self._on_status is not None:
            self._on_status(msg)

    @staticmethod
    def _emit_log(events: "queue.Queue[ASREvent]", msg: str) -> None:
        try:
            events.put_nowait(ASREvent(type="log", source="engine", text=str(msg)))
        except queue.Full:
            pass

    @staticmethod
    def _emit_fatal(events: "queue.Queue[ASREvent]", msg: str) -> None:
        try:
            events.put(ASREvent(type="fatal", source="engine", text=str(msg)), timeout=0.5)
        except queue.Full:
            pass


def _chunks_from(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    """Re-attach an already-pulled first chunk. Must stay a real generator:
    `generate()` switches to streaming mode on `isinstance(x, GeneratorType)`,
    which `itertools.chain` would fail."""
    yield first
    yield from rest


class _WordStreamer:
    """
    `generate(streamer=...)` sink: turns per-step tokens into committed words.

    Streaming `generate()` returns only when the audio stream ends, so the
    result object is useless to us - the streamer is the only live output.
    That costs us the `durations` tensor, so we track the encoder frame index
    the same way the RNN-T decoding loop does: a blank emission advances one
    frame, a non-blank stays put, and `max_symbols_per_step` consecutive
    non-blanks force an advance. Cumulative frames * 80 ms is exactly the
    timestamp `processor.decode(durations=...)` would have produced, and it is
    global across chunks - no `offset_sec` bookkeeping.
    """

    def __init__(
        self,
        *,
        model,
        processor,
        assembler: WordAssembler,
        blank_ids: set,
        blank_id: int,
        on_words: Callable[[List[Word]], None],
    ) -> None:
        from tokenizers.decoders import DecodeStream

        self._assembler = assembler
        self._blank_ids = blank_ids
        self._blank_id = blank_id
        self._on_words = on_words
        self._max_symbols = int(getattr(model.config, "max_symbols_per_step", 10))

        # Same incremental decoder the processor uses for offline timestamps;
        # it is what keeps subword pieces (and their leading spaces) intact.
        self._tokenizer = processor.tokenizer._tokenizer
        self._stream = DecodeStream(skip_special_tokens=True)

        self._frame = 0
        self._symbols = 0
        self._started = False
        self._last_token_at = time.perf_counter()

    def put(self, value) -> None:
        if not self._started:
            # First call carries the decoder's start tokens, not output.
            self._started = True
            return

        self._last_token_at = time.perf_counter()
        token = int(value.reshape(-1)[0])
        if token == self._blank_id:
            self._frame += 1
            self._symbols = 0
            self._emit(self._assembler.advance(self._frame))
            return

        if token not in self._blank_ids:
            piece = self._stream.step(self._tokenizer, token)
            if piece:
                self._emit(self._assembler.push(piece, self._frame))

        self._symbols += 1
        if self._symbols >= self._max_symbols:
            self._frame += 1
            self._symbols = 0

    def flush_if_stale(self, max_idle_sec: float) -> None:
        """Close a pending word when the audio stream itself has gone quiet.

        Called from the mel-chunk generator, i.e. on the same thread as `put`.
        """
        if time.perf_counter() - self._last_token_at >= max_idle_sec:
            self._emit(self._assembler.flush())

    def end(self) -> None:
        self._emit(self._assembler.flush())

    def _emit(self, words: List[Word]) -> None:
        if words:
            self._on_words(words)
