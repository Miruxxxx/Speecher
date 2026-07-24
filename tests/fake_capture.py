"""Stand-in for native/audio_capture: same protocol, no audio hardware.

Used by tests/test_rust_capture.py so the supervisor can be exercised on a
machine with no sound card (and in a plain `pytest tests/` run). The mode comes
from the FAKE_CAPTURE_MODE environment variable:

  stream   info line, then 10 chunks of 100 samples at 0.5, then idle
  trickle  info line, then one sample split across two writes (3 bytes + 1)
  fatal    fatal line, exit 1
  garbage  a non-JSON stderr line, then one chunk
  silent   info line only, no audio at all (what an idle loopback looks like)

Every mode ends by waiting for stdin to close, exactly like the real binary.
"""

import json
import os
import struct
import sys
import time


def emit(**payload):
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.stderr.flush()


def info():
    emit(
        kind="info",
        device="Fake Endpoint",
        source="loopback",
        input_sample_rate=48000,
        input_channels=2,
        input_format="f32",
        output_sample_rate=16000,
        output_channels=1,
    )


def write_samples(values):
    sys.stdout.buffer.write(struct.pack(f"<{len(values)}f", *values))
    sys.stdout.buffer.flush()


def wait_for_stdin_close():
    while sys.stdin.buffer.read(64):
        pass


def main() -> int:
    mode = os.environ.get("FAKE_CAPTURE_MODE", "stream")

    if mode == "fatal":
        emit(kind="fatal", text="no loopback endpoint")
        return 1

    if mode == "garbage":
        sys.stderr.write("thread 'main' panicked at src/main.rs:1:1\n")
        sys.stderr.flush()
        info()
        write_samples([0.25] * 100)
    elif mode == "trickle":
        info()
        # One sample split across two writes: the parent must carry the tail.
        raw = struct.pack("<f", 0.75)
        sys.stdout.buffer.write(raw[:3])
        sys.stdout.buffer.flush()
        time.sleep(0.2)
        sys.stdout.buffer.write(raw[3:])
        sys.stdout.buffer.flush()
    elif mode == "silent":
        info()
    else:
        info()
        for _ in range(10):
            write_samples([0.5] * 100)
            time.sleep(0.01)

    wait_for_stdin_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
