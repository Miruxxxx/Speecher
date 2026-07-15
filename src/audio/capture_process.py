from __future__ import annotations

"""Child-process entry point for WASAPI loopback capture.

PortAudio (pyaudiowpatch) corrupts the process heap after minutes of an idle
loopback stream (docs/CODE_REVIEW_2026-07-15.md §B1), so the whole PortAudio
stack lives in this disposable process. It ships raw float32 frames to the
parent over a multiprocessing queue and owns no other state: when it dies,
the supervisor just starts a new one.

Messages on the queue (kind, payload):
  ("info",  {"name", "index", "channels", "sample_rate"})  once per start
  ("data",  bytes)   raw interleaved float32 frames
  ("fatal", str)     unrecoverable setup error (no device, open failed)
"""

import queue as _queue


def run_capture(
    data_queue,
    stop_event,
    device_hint: str,
    frames_per_buffer: int,
    max_channels: int,
) -> None:
    # Imports happen in the child: keep the parent free of PortAudio entirely.
    import pyaudiowpatch as pyaudio

    from audio.devices import pick_loopback_device

    pa = pyaudio.PyAudio()
    stream = None
    try:
        try:
            idx, name, ch, sr = pick_loopback_device(pa, device_hint)
        except Exception as exc:
            data_queue.put(("fatal", f"pick_loopback_device failed: {exc!r}"))
            return

        channels = max(1, min(int(ch), int(max_channels)))
        src_sr = int(sr)
        data_queue.put(
            ("info", {"name": name, "index": idx, "channels": channels, "sample_rate": src_sr})
        )

        try:
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=src_sr,
                input=True,
                input_device_index=idx,
                frames_per_buffer=int(frames_per_buffer),
            )
        except Exception as exc:
            data_queue.put(("fatal", f"stream open failed: {exc!r}"))
            return

        # NOTE: read() blocks while the system renders no audio (loopback is
        # silent), so stop_event may go unchecked for long stretches. Shutdown
        # is therefore parent-driven: the supervisor terminates this process.
        while not stop_event.is_set():
            raw = stream.read(int(frames_per_buffer), exception_on_overflow=False)
            try:
                data_queue.put(("data", raw), timeout=1.0)
            except _queue.Full:
                # Parent gone or stalled; drop frames rather than deadlock.
                pass
    except Exception as exc:
        try:
            data_queue.put(("fatal", f"capture loop failed: {exc!r}"), timeout=1.0)
        except Exception:
            pass
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        try:
            pa.terminate()
        except Exception:
            pass
