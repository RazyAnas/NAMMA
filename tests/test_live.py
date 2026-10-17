import json
import struct

import cv2
import numpy as np
import pytest

from app.services.live_protocol import parse_packet


def _packet(header: dict, image: np.ndarray) -> bytes:
    encoded = json.dumps(header).encode()
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    return struct.pack("<I", len(encoded)) + encoded + jpeg.tobytes()


def test_parse_packet_round_trip():
    image = np.full((48, 64, 3), 120, np.uint8)
    header, decoded = parse_packet(_packet({"t": 1.5, "gravity": [0, 1, 0]}, image), 1 << 20)
    assert header["t"] == 1.5
    assert decoded.shape == (48, 64, 3)


@pytest.mark.parametrize("data", [b"", b"\x00\x00", struct.pack("<I", 4) + b"{}" + b"x" * 10, struct.pack("<I", 2) + b"[]" + b"notjpeg"])
def test_parse_packet_rejects_malformed(data):
    with pytest.raises(ValueError):
        parse_packet(data, 1 << 20)


def test_parse_packet_enforces_size_limit():
    with pytest.raises(ValueError):
        parse_packet(_packet({}, np.zeros((480, 640, 3), np.uint8)), 100)
