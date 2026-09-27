"""Тесты обработчиков бота и FSM (`app/bot/handlers.py`)."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.bot import handlers as handlers_module
from app.bot.handlers import register_handlers
from app.bot.keyboards import ACTION_OPTIONS, STYLE_OPTIONS
from app.bot.states import StoryFlow
from app.config import Settings
from app.db.models import GenerationStatus, GenerationType
from app.db.repository import GenerationRepository
from app.services.storage import PhotoStorage
from app.services.text_generator import TextGenerationError
from app.services.video_provider import VideoFailedError
from tests.bot_harness import (
    FakeBotStarted,
    FakeDispatcher,
    FakeMessage,
    FakeMessageCallback,
    FakeMessageCreated,
    FakeTextGenerator,
    FakeVideoProvider,
    FakeContext,
    keyboard_labels,
    make_image_attachment,
    make_services,
)
from tests.factories import make_photo_bytes, make_png

TMP_ROOT = Path(".tmp/tests")


@pytest.fixture
def storage() -> PhotoStorage:
    return PhotoStorage(TMP_ROOT / f"uploads-{uuid4().hex}")


def _register(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
    *,
    text_generator=None,
    video_provider=None,
    download_photo=None,
):
    services = make_services(
        settings=settings,
        repository=repository,
        storage=storage,
        text_generator=text_generator,
        video_provider=video_provider,
        download_photo=download_photo,
    )
    dispatcher = FakeDispatcher()
    registered = register_handlers(dispatcher, services)
    return registered, services


# --- Задача 6.1: клавиатуры --------------------------------------------------


def test_keyboards_return_expected_labels() -> None:
    """Построители клавиатур возвращают точные ожидаемые подписи."""

    from app.bot.keyboards import (
        build_action_keyboard,
        build_style_keyboard,
        build_upload_photo_keyboard,
    )

    assert keyboard_labels([build_style_keyboard().as_markup()]) == [
        [style] for style in STYLE_OPTIONS
    ]
    assert keyboard_labels([build_action_keyboard().as_markup()]) == [
        [action] for action in ACTION_OPTIONS
    ]
    assert keyboard_labels([build_upload_photo_keyboard().as_markup()]) == [
        ["Загрузить фото игрушки"]
    ]


# --- Задача 6.2: /start ------------------------------------------------------


async def test_start_sends_greeting_and_photo_button(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """`/start` отправляет приветствие, инструкцию и кнопку загрузки фото."""

    registered, _ = _register(settings, repository, storage)
    event = FakeMessageCreated(FakeMessage())
    context = FakeContext()

    await registered.start(event, context)

    assert event.message.sent_texts == [handlers_module.GREETING]
    assert keyboard_labels(event.message.sent[0].attachments) == [
        ["Загрузить фото игрушки"]
    ]
    assert context.state == StoryFlow.waiting_photo


# --- Задача 6.3: приём фото --------------------------------------------------


async def test_bot_started_sends_greeting_and_photo_button(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Нажатие «Старт» в MAX (`bot_started`) тоже шлёт приветствие и кнопку."""

    registered, _ = _register(settings, repository, storage)
    event = FakeBotStarted(user_id=42, chat_id=42)
    context = FakeContext()

    await registered.bot_started(event, context)

    assert event.sent_texts == [handlers_module.GREETING]
    assert keyboard_labels(event.sent[0].attachments) == [
        ["Загрузить фото игрушки"]
    ]
    assert context.state == StoryFlow.waiting_photo


async def test_callback_is_acknowledged_via_answer(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Колбэк подтверждается через `answer` (пустой `ack` MAX отвергает)."""

    registered, _ = _register(settings, repository, storage)
    event = FakeMessageCallback("upload_photo", message=FakeMessage())
    context = FakeContext()

    await registered.callback(event, context)

    assert event.ack_count == 1
    assert event.answer_attachments == [None]  # ответ собран с сообщением
    assert event.sent_texts == [handlers_module.PHOTO_PROMPT]
    assert context.state == StoryFlow.waiting_photo


async def test_photo_accepts_and_moves_to_style_selection(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Корректное фото сохраняется, пишется в БД и ведёт к выбору параметров."""

    photo = make_photo_bytes(400, 400)

    async def download(url: str) -> bytes:
        return photo

    registered, _ = _register(
        settings, repository, storage, download_photo=download
    )
    event = FakeMessageCreated(
        FakeMessage(attachments=[make_image_attachment("http://max/photo")])
    )
    context = FakeContext(state=StoryFlow.waiting_photo)

    await registered.photo(event, context)

    assert context.state == StoryFlow.waiting_style
    assert keyboard_labels(event.message.sent[-1].attachments) == [
        [style] for style in STYLE_OPTIONS
    ]

    history = await repository.get_history(1)
    assert len(history) == 1
    assert history[0].type is GenerationType.TALE
    assert history[0].local_path is not None
    assert Path(history[0].local_path).read_bytes() == photo
    assert context.data["tale_id"] == history[0].id


async def test_photo_rejects_invalid_dimensions(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Некорректное фото отклоняется с указанием требования."""

    async def download(url: str) -> bytes:
        return make_png(200, 200)

    registered, _ = _register(
        settings, repository, storage, download_photo=download
    )
    event = FakeMessageCreated(
        FakeMessage(attachments=[make_image_attachment("http://max/photo")])
    )
    context = FakeContext(state=StoryFlow.waiting_photo)

    await registered.photo(event, context)

    assert context.state == StoryFlow.waiting_photo
    assert "300" in (event.message.sent_texts[-1] or "")
    assert await repository.get_history(1) == []


async def test_photo_accepts_image_like_attachment(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Вложение распознаётся по типу, а не по классу модели."""

    async def download(url: str) -> bytes:
        return make_photo_bytes(400, 400)

    registered, _ = _register(
        settings, repository, storage, download_photo=download
    )
    attachment = SimpleNamespace(
        type="image",
        payload=SimpleNamespace(url="http://max/photo"),
    )
    event = FakeMessageCreated(FakeMessage(attachments=[attachment]))
    context = FakeContext(state=StoryFlow.waiting_photo)

    await registered.photo(event, context)

    assert context.state == StoryFlow.waiting_style


async def test_photo_rejects_non_image_attachment(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Вложение, не являющееся изображением, отклоняется."""

    registered, _ = _register(settings, repository, storage)
    event = FakeMessageCreated(
        FakeMessage(attachments=[SimpleNamespace(payload=None)])
    )
    context = FakeContext(state=StoryFlow.waiting_photo)

    await registered.photo(event, context)

    assert event.message.sent_texts == [handlers_module.NOT_A_PHOTO]


async def test_text_message_in_photo_state_asks_for_photo(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Текстовое сообщение вместо фото просит прислать фото."""

    registered, _ = _register(settings, repository, storage)
    event = FakeMessageCreated(FakeMessage())
    context = FakeContext(state=StoryFlow.waiting_photo)

    await registered.other_message(event, context)

    assert event.message.sent_texts == [handlers_module.NOT_A_PHOTO]


# --- Задача 6.4: выбор параметров и оркестрация ------------------------------


async def test_style_callback_moves_to_action(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Выбор стиля запрашивает действие без лишних вопросов."""

    registered, _ = _register(settings, repository, storage)
    event = FakeMessageCallback("style:0")
    context = FakeContext(state=StoryFlow.waiting_style)

    await registered.callback(event, context)

    assert context.state == StoryFlow.waiting_action
    assert context.data["style"] == STYLE_OPTIONS[0]
    assert event.sent_texts == [handlers_module.ACTION_PROMPT]
    assert keyboard_labels(event.sent[-1].attachments) == [
        [action] for action in ACTION_OPTIONS
    ]


async def test_full_generation_order_and_artifacts(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Полный цикл отправляет сказку и видео в ожидаемом порядке."""

    photo = make_photo_bytes(400, 400)

    async def download(url: str) -> bytes:
        return photo

    text_generator = FakeTextGenerator(result="Сказка про зайца.")
    video_provider = FakeVideoProvider(
        video_url="https://cdn.example.com/toy.mp4"
    )
    registered, _ = _register(
        settings,
        repository,
        storage,
        text_generator=text_generator,
        video_provider=video_provider,
        download_photo=download,
    )

    photo_event = FakeMessageCreated(
        FakeMessage(attachments=[make_image_attachment("http://max/photo")])
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await registered.photo(photo_event, context)
    photo_path = context.data["photo_path"]

    await registered.callback(
        FakeMessageCallback("style:1", message=FakeMessage()),
        context,
    )
    action_callback = FakeMessageCallback("action:1", message=FakeMessage())
    await registered.callback(action_callback, context)

    assert action_callback.sent_texts == [
        handlers_module.TALE_PROGRESS,
        "Сказка про зайца.",
        handlers_module.VIDEO_PROGRESS,
        "Ваше видео готово:\nhttps://cdn.example.com/toy.mp4",
    ]
    assert text_generator.calls == [(STYLE_OPTIONS[1], ACTION_OPTIONS[1])]
    assert video_provider.created_with[0][1] == ACTION_OPTIONS[1]

    saved = Path(photo_path).read_bytes()
    assert video_provider.created_with[0][0] == base64.b64encode(saved).decode()

    history = await repository.get_history(1)
    by_type = {item.type: item for item in history}
    assert by_type[GenerationType.TALE].status is GenerationStatus.SUCCESS
    assert by_type[GenerationType.TALE].tale_text == "Сказка про зайца."
    assert by_type[GenerationType.VIDEO].status is GenerationStatus.SUCCESS
    assert (
        by_type[GenerationType.VIDEO].video_url
        == "https://cdn.example.com/toy.mp4"
    )
    assert by_type[GenerationType.VIDEO].kling_request_id == "req-fake"


async def test_second_generation_is_blocked_with_subscription(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Вторая генерация блокируется и предлагает подписку."""

    tale = await repository.create_tale(1)
    await repository.update_result(
        tale.id, status=GenerationStatus.SUCCESS, tale_text="Старая сказка"
    )
    video = await repository.create_video(1, request_id="old")
    await repository.update_result(
        video.id, status=GenerationStatus.SUCCESS, video_url="https://old"
    )

    text_generator = FakeTextGenerator()
    photo = make_photo_bytes(400, 400)

    async def download(url: str) -> bytes:
        return photo

    registered, _ = _register(
        settings,
        repository,
        storage,
        text_generator=text_generator,
        download_photo=download,
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_context(registered, context)

    action_callback = FakeMessageCallback("action:0", message=FakeMessage())
    await registered.callback(action_callback, context)

    assert text_generator.calls == []
    assert action_callback.sent_texts == [handlers_module.FREE_LIMIT_REACHED]
    assert keyboard_labels(action_callback.sent[-1].attachments) == [
        ["Оформить подписку"]
    ]

    stub_callback = FakeMessageCallback("subscribe", message=FakeMessage())
    await registered.callback(stub_callback, context)
    assert stub_callback.sent_texts == [handlers_module.SUBSCRIPTION_STUB]


# --- Задача 6.6: восстановление после ошибок ---------------------------------


async def _prepare_context(
    registered,
    context: FakeContext,
) -> None:
    """Провести FSM через приём фото и выбор стиля до состояния действия."""

    await registered.photo(
        FakeMessageCreated(
            FakeMessage(attachments=[make_image_attachment("http://max/p")])
        ),
        context,
    )
    await registered.callback(
        FakeMessageCallback("style:0", message=FakeMessage()),
        context,
    )


async def test_tale_failure_reports_and_offers_restart(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Сбой сказки сообщается пользователю и предлагает начать заново."""

    async def download(url: str) -> bytes:
        return make_photo_bytes(400, 400)

    text_generator = FakeTextGenerator(
        error=TextGenerationError("deepseek down")
    )
    registered, _ = _register(
        settings,
        repository,
        storage,
        text_generator=text_generator,
        download_photo=download,
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_context(registered, context)

    action_callback = FakeMessageCallback("action:0", message=FakeMessage())
    await registered.callback(action_callback, context)

    assert action_callback.sent_texts == [
        handlers_module.TALE_PROGRESS,
        handlers_module.TALE_FAILED,
    ]
    assert keyboard_labels(action_callback.sent[-1].attachments) == [
        ["Начать заново"]
    ]
    history = await repository.get_history(1)
    assert history[0].status is GenerationStatus.FAILED
    assert history[0].error == "deepseek down"


async def test_video_failure_keeps_tale(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Сбой видео после успешной сказки сохраняет сказку доступной."""

    async def download(url: str) -> bytes:
        return make_photo_bytes(400, 400)

    video_provider = FakeVideoProvider(
        wait_error=VideoFailedError("kling failed")
    )
    registered, _ = _register(
        settings,
        repository,
        storage,
        text_generator=FakeTextGenerator(result="Сказка."),
        video_provider=video_provider,
        download_photo=download,
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_context(registered, context)

    action_callback = FakeMessageCallback("action:0", message=FakeMessage())
    await registered.callback(action_callback, context)

    assert action_callback.sent_texts == [
        handlers_module.TALE_PROGRESS,
        "Сказка.",
        handlers_module.VIDEO_PROGRESS,
        handlers_module.VIDEO_FAILED,
    ]
    history = await repository.get_history(1)
    by_type = {item.type: item for item in history}
    assert by_type[GenerationType.TALE].status is GenerationStatus.SUCCESS
    assert by_type[GenerationType.VIDEO].status is GenerationStatus.FAILED

    # Сказка не расходует квоту без видео.
    assert await repository.has_completed_full_generation(1) is False


async def test_video_unavailable_when_photo_missing(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Недоступный файл фото делает шаг с видео недоступным без запроса."""

    registered, _ = _register(
        settings,
        repository,
        storage,
        text_generator=FakeTextGenerator(result="Сказка."),
    )
    context = FakeContext(
        state=StoryFlow.waiting_action,
        data={
            "style": STYLE_OPTIONS[0],
            "tale_id": 1,
            "photo_path": str(TMP_ROOT / "missing-photo.png"),
        },
    )
    await repository.create_tale(1)  # чтобы tale_id существовал

    action_callback = FakeMessageCallback("action:0", message=FakeMessage())
    await registered.callback(action_callback, context)

    assert action_callback.sent_texts == [
        handlers_module.TALE_PROGRESS,
        "Сказка.",
        handlers_module.VIDEO_PROGRESS,
        handlers_module.VIDEO_UNAVAILABLE,
    ]
