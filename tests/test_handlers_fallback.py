"""Fallback-маршрут для сообщений вне FSM-состояния.

Баг: состояние диалога живёт в памяти (`MemoryContext`), поэтому после
перезапуска бота у чата нет состояния. Все маршруты `message_created`, кроме
`/start`, были зарегистрированы с фильтром по состоянию, и первое сообщение
пользователя («привет») попадало в «Проигнорировано»: бот молчал.

Ожидаемое поведение: сообщение без состояния получает то же приветствие с
кнопкой загрузки фото, что и `/start`, а диалог начинается заново.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.bot import handlers as handlers_module
from app.bot.handlers import register_handlers
from app.bot.keyboards import UPLOAD_PHOTO_PAYLOAD
from app.bot.states import StoryFlow
from app.config import Settings
from app.db.repository import GenerationRepository
from app.services.storage import PhotoStorage
from tests.bot_harness import (
    FakeContext,
    FakeDispatcher,
    FakeMessage,
    FakeMessageCreated,
    make_image_attachment,
    make_services,
)

TMP_ROOT = Path(".tmp/tests")


@pytest.fixture
def storage() -> PhotoStorage:
    return PhotoStorage(TMP_ROOT / f"uploads-{uuid4().hex}")


def _dispatcher_with_handlers(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
):
    """Собрать обработчики и вернуть их вместе с диспетчером-заглушкой."""

    services = make_services(
        settings=settings,
        repository=repository,
        storage=storage,
    )
    dispatcher = FakeDispatcher()
    registered = register_handlers(dispatcher, services)
    return registered, dispatcher


async def test_text_without_state_starts_dialog_with_greeting(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Текст в пустом состоянии получает приветствие, а не молчание."""

    registered, _ = _dispatcher_with_handlers(settings, repository, storage)
    event = FakeMessageCreated(FakeMessage())
    context = FakeContext()

    await registered.other_message(event, context)

    assert event.message.sent_texts == [handlers_module.GREETING]
    markup = event.message.sent[0].attachments[0]
    buttons = markup.payload.buttons
    assert [
        (button.text, button.payload) for row in buttons for button in row
    ] == [("Загрузить фото игрушки", UPLOAD_PHOTO_PAYLOAD)]
    assert context.state == StoryFlow.waiting_photo


async def test_photo_without_state_answers_greeting_and_downloads_nothing(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Фото без состояния не теряется: пользователь видит, как начать.

    Скачивание не выполняется: приём фото — отдельный шаг диалога, и в
    пустом состоянии он не начат. Заглушка `download_photo` в
    `make_services` падает при вызове, поэтому тест ловит лишний HTTP.
    """

    registered, _ = _dispatcher_with_handlers(settings, repository, storage)
    event = FakeMessageCreated(
        FakeMessage(attachments=[make_image_attachment("https://example/x.jpg")])
    )
    context = FakeContext()

    await registered.other_message(event, context)

    assert event.message.sent_texts == [handlers_module.GREETING]
    assert context.state == StoryFlow.waiting_photo


async def test_other_message_route_has_no_state_filter_and_is_last(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """У fallback-маршрута нет фильтров, и он зарегистрирован последним.

    Первый подходящий обработчик выигрывает (так работает dispatcher), а
    `/start` и маршруты с состоянием регистрируются раньше: без этого
    порядка fallback перехватывал бы их.
    """

    registered, dispatcher = _dispatcher_with_handlers(
        settings, repository, storage
    )
    registrations = dispatcher.message_created_registrations

    fallback_indexes = [
        index
        for index, (func, args, _) in enumerate(registrations)
        if func is registered.other_message and not args
    ]

    assert fallback_indexes, "нет маршрута без состояния для прочих сообщений"
    assert fallback_indexes == [len(registrations) - 1]


async def test_text_in_age_state_still_shows_age_hint(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Подсказки внутри диалога не подменяются приветствием."""

    registered, _ = _dispatcher_with_handlers(settings, repository, storage)
    event = FakeMessageCreated(FakeMessage())
    context = FakeContext(state=StoryFlow.waiting_age)

    await registered.other_message(event, context)

    assert event.message.sent_texts == [handlers_module.AGE_HINT]
    assert context.state == StoryFlow.waiting_age
