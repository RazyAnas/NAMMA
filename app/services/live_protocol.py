"""Wire format of live-capture keyframe packets sent by the phone page."""

import json
import struct

import cv2
import numpy as np


def parse_packet(data: bytes, max_bytes: int) -> tuple[dict, np.ndarray]:
    """Packet = uint32 little-endian header length, UTF-8 JSON header, JPEG bytes."""
    if len(data) < 4 or len(data) > max_bytes:
        raise ValueError("Packet size out of range.")
    (header_length,) = struct.unpack("<I", data[:4])
    if header_length > 16384 or 4 + header_length >= len(data):
        raise ValueError("Malformed packet header.")
    header = json.loads(data[4 : 4 + header_length].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Packet header must be a JSON object.")
    image = cv2.imdecode(np.frombuffer(data[4 + header_length :], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Keyframe is not a decodable JPEG.")
    return header, image
