"""Фабрики тестовых изображений (без внешних зависимостей)."""

from __future__ import annotations

import struct

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8"


def make_png(width: int, height: int, *, pad: int = 0) -> bytes:
    """Минимальный PNG с корректным IHDR заданных размеров."""

    ihdr_data = struct.pack(">II", width, height) + bytes([8, 2, 0, 0, 0])
    ihdr = (
        struct.pack(">I", len(ihdr_data))
        + b"IHDR"
        + ihdr_data
        + b"\x00\x00\x00\x00"
    )
    return _PNG_SIGNATURE + ihdr + b"\x00" * pad


def make_jpeg(width: int, height: int, *, pad: int = 0) -> bytes:
    """Минимальный JPEG с корректным SOF0 заданных размеров."""

    sof_payload = (
        bytes([8])  # precision
        + struct.pack(">HH", height, width)
        + bytes([3])  # число компонент
        + bytes([1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0])
    )
    sof = b"\xff\xc0" + struct.pack(">H", len(sof_payload) + 2) + sof_payload
    return _JPEG_SIGNATURE + sof + b"\xff\xd9" + b"\x00" * pad


def make_webp(width: int, height: int, *, pad: int = 0) -> bytes:
    """Минимальный WebP (вариант VP8X) с заданным размером канвы."""

    payload = (
        bytes([0x20, 0, 0, 0])
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )
    chunk = b"VP8X" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk + b"\x00" * pad


def make_webp_vp8l(width: int, height: int) -> bytes:
    """WebP без потерь (VP8L)."""

    packed = (width - 1) | ((height - 1) << 14)
    payload = b"\x2f" + packed.to_bytes(4, "little")
    chunk = b"VP8L" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def make_webp_vp8(width: int, height: int) -> bytes:
    """WebP с потерями (VP8 )."""

    frame = (
        b"\x00\x00\x00"
        + b"\x9d\x01\x2a"
        + struct.pack("<H", width)
        + struct.pack("<H", height)
    )
    chunk = b"VP8 " + struct.pack("<I", len(frame)) + frame
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def make_photo_bytes(
    width: int = 400,
    height: int = 400,
    *,
    fmt: str = "png",
    pad: int = 0,
) -> bytes:
    """Фото заданного формата и размеров."""

    if fmt == "png":
        return make_png(width, height, pad=pad)
    if fmt == "jpeg":
        return make_jpeg(width, height, pad=pad)
    if fmt == "webp":
        return make_webp(width, height, pad=pad)
    raise ValueError(f"unsupported format: {fmt}")
