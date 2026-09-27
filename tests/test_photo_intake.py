"""Тесты приёма фото: валидация, сохранение и base64."""

from __future__ import annotations

import base64
from pathlib import Path
from uuid import uuid4

import pytest

from app.config import Settings
from app.services.photo_validation import PhotoValidationError, PhotoValidator
from app.services.storage import PhotoStorage, PhotoStorageError
from tests.factories import (
    make_photo_bytes,
    make_png,
    make_webp,
    make_webp_vp8,
    make_webp_vp8l,
)

TMP_ROOT = Path(".tmp/tests")


# --- Задача 3.2: валидация ---------------------------------------------------


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp"])
async def test_accepts_valid_photo(
    settings: Settings,
    fmt: str,
) -> None:
    """Корректное JPG/PNG/WebP принимается и отдаёт свои размеры."""

    data = make_photo_bytes(400, 400, fmt=fmt)
    info = PhotoValidator(settings).validate(data)

    assert info.format == fmt
    assert (info.width, info.height) == (400, 400)
    assert info.extension in {".png", ".jpg", ".webp"}


@pytest.mark.parametrize(
    "builder",
    [make_webp, make_webp_vp8l, make_webp_vp8],
)
async def test_reads_webp_dimensions(
    settings: Settings,
    builder,
) -> None:
    """Размеры читаются у всех вариантов WebP (VP8X/VP8L/VP8)."""

    info = PhotoValidator(settings).validate(builder(720, 718))

    assert info.format == "webp"
    assert (info.width, info.height) == (720, 718)
    assert info.extension == ".webp"


async def test_rejects_unsupported_format(settings: Settings) -> None:
    """Не-JPG/PNG отклоняется с причиной «формат»."""

    with pytest.raises(PhotoValidationError) as exc:
        PhotoValidator(settings).validate(b"GIF89a" + b"\x00" * 100)

    assert exc.value.reason == "format"
    assert "JPG" in exc.value.user_message


async def test_rejects_oversized_file(settings: Settings) -> None:
    """Файл больше 10 МБ отклоняется с причиной «размер»."""

    oversized = make_png(400, 400, pad=10 * 1024 * 1024)
    with pytest.raises(PhotoValidationError) as exc:
        PhotoValidator(settings).validate(oversized)

    assert exc.value.reason == "size"
    assert "10" in exc.value.user_message


async def test_rejects_too_small_dimensions(settings: Settings) -> None:
    """Изображение меньше 300×300 отклоняется."""

    with pytest.raises(PhotoValidationError) as exc:
        PhotoValidator(settings).validate(make_png(200, 200))

    assert exc.value.reason == "dimensions"
    assert "300" in exc.value.user_message


async def test_rejects_aspect_ratio_out_of_range(settings: Settings) -> None:
    """Слишком вытянутое изображение отклоняется."""

    with pytest.raises(PhotoValidationError) as exc:
        PhotoValidator(settings).validate(make_png(1000, 300))

    assert exc.value.reason == "aspect"


@pytest.mark.parametrize(
    ("width", "height"),
    [(300, 750), (750, 300), (600, 300), (300, 600), (600, 600)],
)
async def test_accepts_aspect_ratio_boundaries(
    settings: Settings,
    width: int,
    height: int,
) -> None:
    """Граничные и внутридиапазонные соотношения сторон принимаются."""

    info = PhotoValidator(settings).validate(make_png(width, height))
    assert (info.width, info.height) == (width, height)


async def test_rejects_empty_file(settings: Settings) -> None:
    """Пустой файл отклоняется."""

    with pytest.raises(PhotoValidationError) as exc:
        PhotoValidator(settings).validate(b"")

    assert exc.value.reason == "empty"


async def test_rejects_corrupted_png(settings: Settings) -> None:
    """Обрезанный PNG не даёт прочитать размеры."""

    with pytest.raises(PhotoValidationError) as exc:
        PhotoValidator(settings).validate(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4)

    assert exc.value.reason == "corrupted"


async def test_jpeg_without_sof_is_corrupted(settings: Settings) -> None:
    """JPEG без SOF-маркера считается повреждённым."""

    with pytest.raises(PhotoValidationError) as exc:
        PhotoValidator(settings).validate(b"\xff\xd8\xff\x00" + b"\x00" * 30)

    assert exc.value.reason == "corrupted"


# --- Задача 3.1: сохранение --------------------------------------------------


async def test_saves_photo_with_expected_name() -> None:
    """Сохранённый файл существует на диске с ожидаемым именем."""

    uploads = TMP_ROOT / f"uploads-{uuid4().hex}"
    storage = PhotoStorage(uploads)
    data = make_png(400, 400)

    path = storage.save(data, extension=".png", user_id=42)

    assert path.exists()
    assert path.parent == uploads
    assert path.name.startswith("42_")
    assert path.suffix == ".png"
    assert path.read_bytes() == data

    for item in uploads.iterdir():
        item.unlink()
    uploads.rmdir()


async def test_save_accepts_extension_without_dot() -> None:
    """Расширение без точки нормализуется."""

    uploads = TMP_ROOT / f"uploads-{uuid4().hex}"
    storage = PhotoStorage(uploads)

    path = storage.save(b"data", extension="jpg", user_id=7)
    assert path.suffix == ".jpg"

    path.unlink()
    uploads.rmdir()


# --- Задача 3.3: base64 для видеопровайдера ----------------------------------


async def test_encodes_saved_photo_to_base64() -> None:
    """Тест проверяет корректную base64-строку из файла."""

    uploads = TMP_ROOT / f"uploads-{uuid4().hex}"
    storage = PhotoStorage(uploads)
    data = make_photo_bytes(400, 400)
    path = storage.save(data, extension=".png", user_id=1)

    encoded = storage.encode_base64(path)

    assert base64.b64decode(encoded) == data

    path.unlink()
    uploads.rmdir()


async def test_encode_missing_file_raises_clear_error() -> None:
    """Ошибка чтения поднимает понятную ошибку, а не пустой результат."""

    storage = PhotoStorage(TMP_ROOT / "missing")

    with pytest.raises(PhotoStorageError) as exc:
        storage.encode_base64(TMP_ROOT / "does-not-exist.png")

    assert exc.value.reason == "read"
    assert exc.value.user_message
