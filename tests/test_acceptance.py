"""Приёмочный end-to-end тест сценария бота (задача 8.2)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.bot import handlers as handlers_module
from app.bot.handlers import register_handlers
from app.bot.keyboards import ACTION_OPTIONS, STYLE_OPTIONS
from app.bot.states import StoryFlow
from app.config import Settings
from app.db.engine import Database
from app.db.models import GenerationStatus, GenerationType
from app.db.repository import GenerationRepository
from app.services.storage import PhotoStorage
from app.services.video_provider import MockVideoProvider, build_video_provider
from tests.bot_harness import (
    FakeDispatcher,
    FakeContext,
    FakeMessage,
    FakeMessageCallback,
    FakeMessageCreated,
    FakeTextGenerator,
    keyboard_labels,
    make_image_attachment,
    make_services,
)
from tests.factories import make_photo_bytes

TMP_ROOT = Path(".tmp/tests")


async def test_end_to_end_acceptance(database: Database) -> None:
    """Полный сценарий: /start → фото → стиль → действие → результат → квота."""

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_bot_token="test-token",
        kling_api_key="",  # включает mock-провайдер
    )

    provider = build_video_provider(settings)
    assert isinstance(provider, MockVideoProvider)

    repository = GenerationRepository(database)
    storage = PhotoStorage(TMP_ROOT / f"e2e-uploads-{uuid4().hex}")
    text_generator = FakeTextGenerator(result="Жила-была плюшевая игрушка.")

    async def download(url: str) -> bytes:
        return make_photo_bytes(400, 400)

    services = make_services(
        settings=settings,
        repository=repository,
        storage=storage,
        text_generator=text_generator,
        video_provider=provider,
        download_photo=download,
    )
    registered = register_handlers(FakeDispatcher(), services)
    context = FakeContext()

    # 1. Приветствие по /start с кнопкой загрузки фото.
    start_event = FakeMessageCreated(FakeMessage())
    await registered.start(start_event, context)
    assert keyboard_labels(start_event.message.sent[0].attachments) == [
        ["Загрузить фото игрушки"]
    ]
    assert context.state == StoryFlow.waiting_photo

    # 2. Корректное фото → предложение стиля.
    await registered.photo(
        FakeMessageCreated(
            FakeMessage(attachments=[make_image_attachment("http://max/p")])
        ),
        context,
    )
    assert context.state == StoryFlow.waiting_style

    # 3. Выбор стиля и действия.
    await registered.callback(
        FakeMessageCallback("style:0", message=FakeMessage()), context
    )
    assert context.state == StoryFlow.waiting_action

    action_callback = FakeMessageCallback("action:2", message=FakeMessage())
    await registered.callback(action_callback, context)

    # 4. Выдача сказки и mock-видео.
    assert action_callback.sent_texts == [
        handlers_module.TALE_PROGRESS,
        "Жила-была плюшевая игрушка.",
        handlers_module.VIDEO_PROGRESS,
        f"Ваше видео готово:\n{settings.mock_video_url}",
    ]

    # 5. Записи появились в истории и квота израсходована.
    history = await repository.get_history(1)
    by_type = {item.type: item for item in history}
    assert by_type[GenerationType.TALE].status is GenerationStatus.SUCCESS
    assert by_type[GenerationType.VIDEO].status is GenerationStatus.SUCCESS
    assert by_type[GenerationType.VIDEO].video_url == settings.mock_video_url
    assert await repository.has_completed_full_generation(1) is True

    # 6. Вторая полная генерация блокируется и предлагает подписку.
    await registered.start(FakeMessageCreated(FakeMessage()), context)
    await registered.photo(
        FakeMessageCreated(
            FakeMessage(attachments=[make_image_attachment("http://max/p")])
        ),
        context,
    )
    await registered.callback(
        FakeMessageCallback("style:1", message=FakeMessage()), context
    )
    blocked = FakeMessageCallback("action:1", message=FakeMessage())
    await registered.callback(blocked, context)

    assert blocked.sent_texts == [handlers_module.FREE_LIMIT_REACHED]
    assert keyboard_labels(blocked.sent[-1].attachments) == [
        ["Оформить подписку"]
    ]

    # 7. Подписка — заглушка в MVP.
    stub = FakeMessageCallback("subscribe", message=FakeMessage())
    await registered.callback(stub, context)
    assert stub.sent_texts == [handlers_module.SUBSCRIPTION_STUB]

    # Параметры сценария были переданы в генератор.
    assert text_generator.calls == [(STYLE_OPTIONS[0], ACTION_OPTIONS[2])]
