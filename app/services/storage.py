"""Сохранение принятых фото и чтение их для фоновой генерации."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4


class PhotoStorageError(Exception):
    """Не удалось сохранить или прочитать фото."""

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

    def read_bytes(self, path: str | Path) -> bytes:
        """Прочитать сохранённый файл по пути.

        Отдельный метод нужен фоновой генерации: пайплайн и распознавание
        работают с байтами фотографии, а не с путём, и читать файл должен
        сервис хранения, а не обработчик события.
        """

        file_path = Path(path)
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            raise PhotoStorageError(
                "Не удалось прочитать сохранённое фото. Пришлите фото ещё раз.",
                reason="read",
            ) from exc
        if not data:
            raise PhotoStorageError(
                "Сохранённое фото пустое. Пришлите фото ещё раз.",
                reason="empty",
            )
        return data


__all__ = ["PhotoStorage", "PhotoStorageError"]
