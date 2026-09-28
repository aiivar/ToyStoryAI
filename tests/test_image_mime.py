"""Тесты определения MIME по сигнатуре (`app/utils/image_mime.py`).

MAX отдаёт загруженные фото транскодированными в WebP, поэтому тип
определяется по первым байтам файла, а не по расширению: JPEG, PNG и WebP
должны распознаваться одинаково надёжно.
"""

from __future__ import annotations

import base64

import pytest

from app.utils.image_mime import (
    UnsupportedImageTypeError,
    detect_image_mime_type,
    image_data_uri,
)

# Минимальные реальные заголовки: подпись файла плюс хвост, который
# отличает WebP (RIFF-контейнер) от прочих RIFF-файлов.
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00"
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
WEBP_BYTES = b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(JPEG_BYTES, "image/jpeg", id="jpeg"),
        pytest.param(PNG_BYTES, "image/png", id="png"),
        pytest.param(WEBP_BYTES, "image/webp", id="webp"),
    ],
)
def test_detects_mime_type_by_signature(data: bytes, expected: str) -> None:
    """JPEG, PNG и WebP распознаются по первым байтам файла."""

    assert detect_image_mime_type(data) == expected


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"GIF89a\x01\x00\x01\x00", id="gif"),
        pytest.param(b"RIFF\x24\x00\x00\x00WAVEfmt ", id="riff-non-webp"),
        pytest.param("не картинка вовсе".encode("utf-8"), id="text"),
        pytest.param(b"RIFF", id="truncated-riff"),
    ],
)
def test_unknown_bytes_are_not_recognized(data: bytes) -> None:
    """Неизвестный формат не выдаётся за поддерживаемый тип."""

    assert detect_image_mime_type(data) is None


@pytest.mark.parametrize(
    ("data", "mime_type"),
    [
        pytest.param(JPEG_BYTES, "image/jpeg", id="jpeg"),
        pytest.param(PNG_BYTES, "image/png", id="png"),
        pytest.param(WEBP_BYTES, "image/webp", id="webp"),
    ],
)
def test_data_uri_carries_detected_mime_type(
    data: bytes, mime_type: str
) -> None:
    """data URI собирается с типом по содержимому и base64-данными."""

    uri = image_data_uri(data)

    assert uri == (
        f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
    )


def test_data_uri_rejects_unknown_bytes() -> None:
    """Неопознанные байты не превращаются в data URI."""

    with pytest.raises(UnsupportedImageTypeError):
        image_data_uri(b"GIF89a\x01\x00\x01\x00")
