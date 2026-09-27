"""Репозиторий истории генераций.

Инкапсулирует операции записи и чтения записей `Generation`, включая
запрос квоты: израсходована ли у пользователя бесплатная полная генерация.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import Database
from app.db.models import Generation, GenerationStatus, GenerationType


class GenerationRepository:
    """Операции над записями о генерациях."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create_tale(
        self,
        user_id: int,
        *,
        status: GenerationStatus = GenerationStatus.IN_PROGRESS,
        local_path: str | None = None,
        tale_text: str | None = None,
    ) -> Generation:
        """Создать запись о сказке (обычно в начале попытки)."""

        return await self._create(
            user_id=user_id,
            generation_type=GenerationType.TALE,
            status=status,
            local_path=local_path,
            tale_text=tale_text,
        )

    async def create_video(
        self,
        user_id: int,
        *,
        request_id: str | None = None,
        status: GenerationStatus = GenerationStatus.IN_PROGRESS,
        local_path: str | None = None,
    ) -> Generation:
        """Создать запись о видео с идентификатором запроса провайдера."""

        return await self._create(
            user_id=user_id,
            generation_type=GenerationType.VIDEO,
            status=status,
            local_path=local_path,
            kling_request_id=request_id,
        )

    async def _create(
        self,
        *,
        user_id: int,
        generation_type: GenerationType,
        status: GenerationStatus,
        local_path: str | None = None,
        tale_text: str | None = None,
        kling_request_id: str | None = None,
    ) -> Generation:
        async with self._db.session_scope() as session:
            generation = Generation(
                user_id=user_id,
                type=generation_type,
                status=status,
                local_path=local_path,
                tale_text=tale_text,
                kling_request_id=kling_request_id,
            )
            session.add(generation)
            await session.commit()
            await session.refresh(generation)
            return generation

    async def update_result(
        self,
        generation_id: int,
        *,
        status: GenerationStatus,
        tale_text: str | None = None,
        video_url: str | None = None,
        error: str | None = None,
    ) -> Generation | None:
        """Обновить статус и результат записи.

        Возвращает обновлённую запись или `None`, если её нет.
        """

        async with self._db.session_scope() as session:
            generation = await session.get(Generation, generation_id)
            if generation is None:
                return None

            generation.status = status
            if tale_text is not None:
                generation.tale_text = tale_text
            if video_url is not None:
                generation.video_url = video_url
            if error is not None:
                generation.error = error

            await session.commit()
            await session.refresh(generation)
            return generation

    async def get_by_id(self, generation_id: int) -> Generation | None:
        """Вернуть запись по идентификатору."""

        async with self._db.session_scope() as session:
            return await session.get(Generation, generation_id)

    async def get_history(
        self,
        user_id: int,
        *,
        limit: int | None = None,
    ) -> Sequence[Generation]:
        """История пользователя от новых записей к старым."""

        statement = (
            select(Generation)
            .where(Generation.user_id == user_id)
            .order_by(Generation.created_at.desc(), Generation.id.desc())
        )
        if limit is not None:
            statement = statement.limit(limit)

        async with self._db.session_scope() as session:
            result = await session.execute(statement)
            return result.scalars().all()

    async def has_completed_full_generation(self, user_id: int) -> bool:
        """Израсходована ли бесплатная полная генерация пользователя.

        Считается израсходованной, если в истории пользователя есть и
        успешно завершённая сказка, и успешно завершённое видео.
        """

        statement = (
            select(Generation.type)
            .where(
                Generation.user_id == user_id,
                Generation.status == GenerationStatus.SUCCESS,
            )
            .distinct()
        )

        async with self._db.session_scope() as session:
            result = await session.execute(statement)
            completed_types = set(result.scalars().all())

        return {
            GenerationType.TALE,
            GenerationType.VIDEO,
        }.issubset(completed_types)


__all__ = ["GenerationRepository"]
