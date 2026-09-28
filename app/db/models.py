"""Модели SQLAlchemy 2.0 для истории генераций.

Одна таблица `generations` хранит по строке на артефакт (сказка или видео)
за попытку генерации: пользователь, время, тип, статус, ссылки на артефакты,
идентификатор запроса видеопровайдера и признак ошибки.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Базовый класс декларативных моделей."""


class GenerationType(str, enum.Enum):
    """Тип артефакта генерации."""

    TALE = "tale"
    VIDEO = "video"


class GenerationStatus(str, enum.Enum):
    """Жизненный цикл записи о генерации."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


def _utcnow() -> datetime:
    """Текущее время в UTC (используется как значение по умолчанию)."""

    return datetime.now(timezone.utc)


class Generation(Base):
    """Запись о генерации одного артефакта (сказки или видео)."""

    __tablename__ = "generations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )
    type: Mapped[GenerationType] = mapped_column(
        Enum(
            GenerationType,
            native_enum=False,
            length=16,
            values_callable=lambda enum_cls: [item.value for item in enum_cls],
        ),
        nullable=False,
    )
    status: Mapped[GenerationStatus] = mapped_column(
        Enum(
            GenerationStatus,
            native_enum=False,
            length=16,
            values_callable=lambda enum_cls: [item.value for item in enum_cls],
        ),
        default=GenerationStatus.IN_PROGRESS,
        nullable=False,
    )

    # Ссылки на артефакты.
    tale_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    local_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Провайдер-специфичные поля.
    kling_request_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Содержимое постраничной книжки.
    pages_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Сколько страниц в выданной сказке; None у старых записей",
    )
    output_dir: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment=(
            "Каталог прогона: диагностическое поле, ссылку в чат не отдаём"
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - отладочное представление
        return (
            f"Generation(id={self.id!r}, user_id={self.user_id!r}, "
            f"type={self.type!r}, status={self.status!r})"
        )


__all__ = [
    "Base",
    "Generation",
    "GenerationStatus",
    "GenerationType",
]
