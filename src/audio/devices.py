from __future__ import annotations

from typing import List, Tuple

import pyaudiowpatch as pyaudio


LoopbackDevice = Tuple[int, str, int, int]


def list_loopbacks(pa: pyaudio.PyAudio) -> List[LoopbackDevice]:
    items: List[LoopbackDevice] = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info.get("name") or ""
        if "loopback" in name.lower():
            items.append(
                (
                    i,
                    name,
                    int(info.get("maxInputChannels") or 0),
                    int(info.get("defaultSampleRate") or 48000),
                )
            )
    return items


def pick_loopback_device(pa: pyaudio.PyAudio, hint: str) -> LoopbackDevice:
    hint_l = (hint or "").lower().strip()
    loopbacks = list_loopbacks(pa)

    if not loopbacks:
        raise RuntimeError("No loopback devices found")

    # Prefer devices that match the hint AND have input channels.
    for item in loopbacks:
        if hint_l and hint_l in item[1].lower() and item[2] > 0:
            return item

    # Fall back to any name-match.
    for item in loopbacks:
        if hint_l and hint_l in item[1].lower():
            return item

    return loopbacks[0]
