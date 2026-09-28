"""Определение MIME-типа изображения по сигнатуре файла.

Провайдер изображений (reference-картинки) и сервис распознавания игрушки
по фото передают байты вендору как data URI, где тип обязан соответствовать
содержимому: MAX отдаёт загруженные фото транскодированными в WebP, а имя
файла после сохранения может быть любым. Поэтому тип определяется по первым
байтам, а не по расширению и не по догадке.

Модуль лежит в `app/utils/`, потому что нужен двум независимым подсистемам —
`app/story_pipeline/providers/image.py` и `app/services/toy_vision.py` — и не
знает ни об одной из них: ни логики пайплайна, ни правил приёма фото.
Зависимостей нет, размер файла не проверяется (это делает приём фото).
"""

from __future__ import annotations

import base64

# MIME-типы, которые понимают внешние API проекта.
MIME_JPEG = "image/jpeg"
MIME_PNG = "image/png"
MIME_WEBP = "image/webp"

# Подписи файлов: JPEG начинается с SOI-маркера, PNG — с фиксированных
# восьми байт, WebP лежит в RIFF-контейнере, у которого байты 8..12 —
# строка "WEBP" (у WAV там "WAVE", поэтому одной проверки "RIFF" мало).
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_RIFF_SIGNATURE = b"RIFF"
_WEBP_TAG = b"WEBP"

# Смещение и длина тега формата внутри RIFF-контейнера.
_RIFF_TAG_OFFSET = 8
_RIFF_TAG_LENGTH = 4


class UnsupportedImageTypeError(ValueError):
    """Байты не распознаны как JPEG, PNG или WebP.

    Наследует `ValueError`, чтобы вызывающий код мог отдельно обработать
    неопознанные данные, не ловя всё подряд.
    """


def detect_image_mime_type(data: bytes) -> str | None:
    """Вернуть MIME-тип по сигнатуре файла; `None` — формат неизвестен."""

    if data.startswith(_PNG_SIGNATURE):
        return MIME_PNG
    if data.startswith(_JPEG_SIGNATURE):
        return MIME_JPEG
    if _is_riff_webp(data):
        return MIME_WEBP
    return None


def image_data_uri(data: bytes) -> str:
    """Собрать data URI с типом, определённым по содержимому байтов.

    Явный data URI не зависит от срока жизни CDN-ссылки и от того, каким
    именем файл оказался на диске.
    """

    mime_type = detect_image_mime_type(data)
    if mime_type is None:
        raise UnsupportedImageTypeError(
            "Не удалось определить формат изображения по первым байтам: "
            "ожидается JPEG, PNG или WebP."
        )
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _is_riff_webp(data: bytes) -> bool:
    """Проверить, что данные — RIFF-контейнер с тегом формата WebP."""

    if not data.startswith(_RIFF_SIGNATURE):
        return False
    tag = data[_RIFF_TAG_OFFSET : _RIFF_TAG_OFFSET + _RIFF_TAG_LENGTH]
    return tag == _WEBP_TAG


__all__ = [
    "MIME_JPEG",
    "MIME_PNG",
    "MIME_WEBP",
    "UnsupportedImageTypeError",
    "detect_image_mime_type",
    "image_data_uri",
]
