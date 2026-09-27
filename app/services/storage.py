"""Сохранение принятых фото и подготовка base64 для видеопровайдера."""

from __future__ import annotations

import base64
from pathlib import Path
from uuid import uuid4


class PhotoStorageError(Exception):
    """Не удалось сохранить фото или закодировать его в base64."""

    def __init__(self, user_message: str, *, reason: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.reason = reason


class PhotoStorage:
    """Локальное хранение фото игрушек в каталоге `uploads/`."""

    def __init__(self, uploads_dir: str | Path) -> None:
        self._uploads_dir = Path(uploads_dir)

    @property
    def uploads_dir(self) -> Path:
        return self._uploads_dir

    def save(
        self,
        data: bytes,
        *,
        extension: str,
        user_id: int,
    ) -> Path:
        """Сохранить фото на диск и вернуть путь к файлу.

        Имя файла: `<user_id>_<uuid><extension>`.
        """

        if not extension.startswith("."):
            extension = f".{extension}"

        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{user_id}_{uuid4().hex}{extension}"
        path = self._uploads_dir / filename

        try:
            path.write_bytes(data)
        except OSError as exc:  # pragma: no cover - редкий сбой диска
            raise PhotoStorageError(
                "Не удалось сохранить фото. Попробуйте прислать его ещё раз.",
                reason="write",
            ) from exc

        return path

    def encode_base64(self, path: str | Path) -> str:
        """Прочитать сохранённый файл и закодировать его в base64.

        Поднимает `PhotoStorageError` при ошибке чтения или кодирования,
        чтобы вызывающий код не отправлял провайдеру некорректный запрос.
        """

        file_path = Path(path)
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            raise PhotoStorageError(
                "Не удалось прочитать сохранённое фото, поэтому видео "
                "сейчас недоступно.",
                reason="read",
            ) from exc

        if not data:
            raise PhotoStorageError(
                "Сохранённое фото пустое, поэтому видео сейчас недоступно.",
                reason="empty",
            )

        try:
            return base64.b64encode(data).decode("ascii")
        except (ValueError, UnicodeError) as exc:  # pragma: no cover
            raise PhotoStorageError(
                "Не удалось закодировать фото для видео.",
                reason="encode",
            ) from exc


__all__ = ["PhotoStorage", "PhotoStorageError"]
