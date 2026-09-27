"""Валидация входящего фото игрушки.

Правила MVP: JPG, PNG или WebP, не более 10 МБ, не менее 300×300 пикселей,
соотношение сторон от 1:2.5 до 2.5:1. WebP нужен, потому что MAX отдаёт
загруженные фото уже транскодированными в WebP. Размеры читаются из
заголовков PNG/JPEG/WebP, поэтому внешняя зависимость не нужна.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_RIFF_SIGNATURE = b"RIFF"
_WEBP_SIGNATURE = b"WEBP"

# SOF-маркеры JPEG, несущие размеры кадра (исключая DHT/JPG/DAC).
_JPEG_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


class PhotoValidationError(Exception):
    """Фото не прошло валидацию.

    `user_message` — готовая формулировка для отправки пользователю.
    """

    def __init__(self, user_message: str, *, reason: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.reason = reason


@dataclass(frozen=True)
class ImageInfo:
    """Результат успешной валидации изображения."""

    format: str  # "jpeg" или "png"
    width: int
    height: int
    size_bytes: int

    @property
    def extension(self) -> str:
        """Расширение файла для сохранения на диск."""

        return {
            "png": ".png",
            "webp": ".webp",
        }.get(self.format, ".jpg")


def _detect_format(data: bytes) -> str | None:
    if data.startswith(_PNG_SIGNATURE):
        return "png"
    if data.startswith(_JPEG_SIGNATURE):
        return "jpeg"
    if (
        len(data) >= 12
        and data.startswith(_RIFF_SIGNATURE)
        and data[8:12] == _WEBP_SIGNATURE
    ):
        return "webp"
    return None


def _png_dimensions(data: bytes) -> tuple[int, int]:
    # IHDR: 8 байт сигнатуры, 4 байта длины, 4 байта "IHDR",
    # затем ширина (4) и высота (4) в big-endian.
    if len(data) < 24:
        raise PhotoValidationError(
            "Файл изображения повреждён. Пришлите другое фото игрушки.",
            reason="corrupted",
        )
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    index = 2
    length = len(data)
    while index + 1 < length:
        if data[index] != 0xFF:
            index += 1
            continue

        # Пропускаем байты-заполнители 0xFF.
        while index < length and data[index] == 0xFF:
            index += 1
        if index >= length:
            break

        marker = data[index]
        index += 1

        # Маркеры без полезной нагрузки.
        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            continue
        if marker == 0xDA:  # начало сжатых данных — дальше размеров нет.
            break

        if index + 1 >= length:
            break
        segment_length = int.from_bytes(data[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > length:
            break

        if marker in _JPEG_SOF_MARKERS:
            # precision(1) + height(2) + width(2)
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            return width, height

        index += segment_length

    raise PhotoValidationError(
        "Не удалось прочитать размеры изображения. Пришлите другое фото.",
        reason="corrupted",
    )


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    """Читает размеры кадра WebP (расширенный VP8X, lossy VP8, lossless VP8L)."""

    if len(data) < 20:
        raise PhotoValidationError(
            "Файл изображения повреждён. Пришлите другое фото игрушки.",
            reason="corrupted",
        )

    chunk = data[12:16]

    if chunk == b"VP8X":
        if len(data) < 30:
            raise PhotoValidationError(
                "Файл изображения повреждён. Пришлите другое фото игрушки.",
                reason="corrupted",
            )
        # flags(1) + reserved(3) + canvas width-1 (3, LE) + height-1 (3, LE)
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height

    if chunk == b"VP8L":
        if len(data) < 25 or data[20] != 0x2F:
            raise PhotoValidationError(
                "Файл изображения повреждён. Пришлите другое фото игрушки.",
                reason="corrupted",
            )
        packed = int.from_bytes(data[21:25], "little")
        width = (packed & 0x3FFF) + 1
        height = ((packed >> 14) & 0x3FFF) + 1
        return width, height

    if chunk == b"VP8 ":
        # frame tag (3) + start code 0x9d 0x01 0x2a (3) + width/height (2+2)
        payload = data[20:30]
        if len(payload) < 10 or payload[3:6] != b"\x9d\x01\x2a":
            raise PhotoValidationError(
                "Файл изображения повреждён. Пришлите другое фото игрушки.",
                reason="corrupted",
            )
        width = int.from_bytes(payload[6:8], "little") & 0x3FFF
        height = int.from_bytes(payload[8:10], "little") & 0x3FFF
        return width, height

    raise PhotoValidationError(
        "Не удалось прочитать размеры WebP-изображения. Пришлите другое фото.",
        reason="corrupted",
    )


class PhotoValidator:
    """Проверяет фото по правилам приёма MVP."""

    def __init__(self, settings: Settings) -> None:
        self._max_bytes = int(settings.photo_max_size_mb * 1024 * 1024)
        self._min_side = settings.photo_min_side_px
        self._aspect_min = settings.photo_aspect_min
        self._aspect_max = settings.photo_aspect_max

    def validate(self, data: bytes) -> ImageInfo:
        """Вернуть сведения об изображении или поднять `PhotoValidationError`."""

        size_bytes = len(data)
        if size_bytes == 0:
            raise PhotoValidationError(
                "Файл пустой. Пришлите другое фото игрушки.",
                reason="empty",
            )

        if size_bytes > self._max_bytes:
            limit_mb = self._max_bytes / (1024 * 1024)
            raise PhotoValidationError(
                f"Фото больше {limit_mb:g} МБ. Пришлите файл меньшего размера.",
                reason="size",
            )

        image_format = _detect_format(data)
        if image_format is None:
            raise PhotoValidationError(
                "Поддерживаются только JPG, PNG и WebP. Пришлите другое фото.",
                reason="format",
            )

        if image_format == "png":
            width, height = _png_dimensions(data)
        elif image_format == "webp":
            width, height = _webp_dimensions(data)
        else:
            width, height = _jpeg_dimensions(data)

        if width < self._min_side or height < self._min_side:
            raise PhotoValidationError(
                f"Слишком маленькое изображение: нужно не менее "
                f"{self._min_side}×{self._min_side} пикселей.",
                reason="dimensions",
            )

        if width == 0 or height == 0:
            raise PhotoValidationError(
                "Не удалось определить размеры фото. Пришлите другое.",
                reason="corrupted",
            )

        aspect = width / height
        if aspect < self._aspect_min or aspect > self._aspect_max:
            low = 1 / self._aspect_max
            high = self._aspect_max
            raise PhotoValidationError(
                "Недопустимое соотношение сторон: допустимо от "
                f"1:{high:g} до {high:g}:1 (текущее ≈ {aspect:.2f}:1).",
                reason="aspect",
            )

        return ImageInfo(
            format=image_format,
            width=width,
            height=height,
            size_bytes=size_bytes,
        )


__all__ = ["ImageInfo", "PhotoValidationError", "PhotoValidator"]
