"""Тесты обработчиков бота и FSM (`app/bot/handlers.py`).

Задачи 5.2, 5.4, 5.7, 5.8: приём фото, вопрос возраста, вопрос морали,
фоновый запуск генерации и защита от повторного платного прогона. Всё
проверяется на фейковых событиях MAX и фейковых провайдерах — живых вызовов
в тестах нет.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.bot import handlers as handlers_module
from app.bot import runner as runner_module
from app.bot.handlers import register_handlers
from app.bot.keyboards import MORAL_OPTIONS
from app.bot.states import StoryFlow
from app.config import Settings
from app.db.models import GenerationStatus, GenerationType
from app.db.repository import GenerationRepository
from app.services.storage import PhotoStorage
from tests.bot_harness import (
    FakeContext,
    FakeDispatcher,
    FakeMessage,
    FakeMessageCallback,
    FakeMessageCreated,
    FakePipeline,
    FakeSender,
    FakeVision,
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
    sender: FakeSender | None = None,
    vision: FakeVision | None = None,
    pipeline: FakePipeline | None = None,
    download_photo=None,
):
    services = make_services(
        settings=settings,
        repository=repository,
        storage=storage,
        sender=sender,
        vision=vision,
        pipeline=pipeline,
        download_photo=download_photo,
    )
    registered = register_handlers(FakeDispatcher(), services)
    return registered, services


def _photo_download(photo: bytes):
    async def download(url: str) -> bytes:
        return photo

    return download


# --- /start и приём фото ------------------------------------------------------


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


def test_greeting_describes_pages_and_never_video() -> None:
    """Приветствие обещает книжку со страницами и не обещает видео.

    Продукт больше не выдаёт видео, поэтому текст должен называть именно то,
    что пользователь получит: сказку с картинками по одной странице в
    сообщении. Проверка ловит возврат старого текста с «видео».
    """

    greeting = handlers_module.GREETING.lower()

    assert "страниц" in greeting
    assert "видео" not in greeting
    # Шаги диалога прежние: фото → возраст → мораль.
    assert "фото" in greeting
    assert "возраст" in greeting
    assert "морал" in greeting


async def test_bot_started_sends_greeting_and_photo_button(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Нажатие «Старт» в MAX (`bot_started`) тоже шлёт приветствие и кнопку."""

    from tests.bot_harness import FakeBotStarted

    registered, _ = _register(settings, repository, storage)
    event = FakeBotStarted(user_id=42, chat_id=42)
    context = FakeContext()

    await registered.bot_started(event, context)

    assert event.sent_texts == [handlers_module.GREETING]
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


async def test_photo_asks_age_and_saves_record(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Корректное фото сохраняется, пишется в БД и запрашивает возраст."""

    photo = make_photo_bytes(400, 400)
    registered, _ = _register(
        settings,
        repository,
        storage,
        download_photo=_photo_download(photo),
    )
    event = FakeMessageCreated(
        FakeMessage(attachments=[make_image_attachment("http://max/photo")])
    )
    context = FakeContext(state=StoryFlow.waiting_photo)

    await registered.photo(event, context)

    assert context.state == StoryFlow.waiting_age
    assert event.message.sent_texts == [handlers_module.AGE_PROMPT]
    assert keyboard_labels(event.message.sent[-1].attachments) == [
        ["3 года"],
        ["4 года"],
        ["5 лет"],
        ["6 лет"],
    ]

    history = await repository.get_history(1)
    assert len(history) == 1
    assert history[0].type is GenerationType.TALE
    assert Path(history[0].local_path).read_bytes() == photo
    assert context.data["tale_id"] == history[0].id


async def test_photo_rejects_invalid_dimensions(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Некорректное фото отклоняется с указанием требования."""

    registered, _ = _register(
        settings,
        repository,
        storage,
        download_photo=_photo_download(make_png(200, 200)),
    )
    event = FakeMessageCreated(
        FakeMessage(attachments=[make_image_attachment("http://max/photo")])
    )
    context = FakeContext(state=StoryFlow.waiting_photo)

    await registered.photo(event, context)

    assert context.state == StoryFlow.waiting_photo
    assert "300" in (event.message.sent_texts[-1] or "")
    assert await repository.get_history(1) == []


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


# --- Диалог: возраст и мораль -------------------------------------------------


async def _prepare_dialog(
    registered,
    context: FakeContext,
    photo: bytes = make_photo_bytes(400, 400),
) -> None:
    """Провести FSM через приём фото до вопроса о возрасте."""

    await registered.photo(
        FakeMessageCreated(
            FakeMessage(attachments=[make_image_attachment("http://max/p")])
        ),
        context,
    )


async def test_age_callback_asks_moral_and_remembers_age(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Выбор возраста сохраняется в FSM и открывает вопрос о морали."""

    registered, _ = _register(
        settings,
        repository,
        storage,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)

    event = FakeMessageCallback("age:5")
    await registered.callback(event, context)

    assert context.state == StoryFlow.waiting_moral
    assert context.data["age"] == 5
    assert event.sent_texts == [handlers_module.MORAL_PROMPT]
    assert keyboard_labels(event.sent[-1].attachments) == [
        [moral] for moral in MORAL_OPTIONS
    ]


async def test_full_dialog_stores_age_and_moral_and_runs_generation(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Полный диалог сохраняет возраст и мораль и передаёт их в генерацию."""

    sender = FakeSender()
    vision = FakeVision()
    pipeline = FakePipeline()
    registered, services = _register(
        settings,
        repository,
        storage,
        sender=sender,
        vision=vision,
        pipeline=pipeline,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)
    await registered.callback(FakeMessageCallback("age:6"), context)

    event = FakeMessageCallback("moral:1")
    await registered.callback(event, context)
    await registered.runs.wait_idle()

    # Возраст и мораль сохранены в состоянии диалога на момент запуска.
    assert any(
        snapshot.get("age") == 6
        and snapshot.get("moral") == MORAL_OPTIONS[1]
        for snapshot in context.history
    )
    assert event.sent_texts == [handlers_module.GENERATION_STARTED]
    # Диалог на этом закончен: состояние снова ждёт фото, а выбранные
    # параметры ушли в прогон, поэтому хранить их дальше незачем.
    assert context.state == StoryFlow.waiting_photo
    assert context.data == {}

    # Фото ушло на распознавание и прогон отчитался пользователю.
    assert vision.photos == [make_photo_bytes(400, 400)]
    assert sender.sent_texts


async def test_moral_value_reaches_pipeline_unchanged(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Выбранная мораль доходит до пайплайна без изменения формулировки."""

    seen: list[object] = []

    class _CapturingPipeline(FakePipeline):
        async def run(self, state):  # type: ignore[no-untyped-def]
            seen.append(state)
            return await super().run(state)

    pipeline = _CapturingPipeline()
    registered, _ = _register(
        settings,
        repository,
        storage,
        pipeline=pipeline,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)
    await registered.callback(FakeMessageCallback("age:4"), context)
    await registered.callback(FakeMessageCallback("moral:2"), context)
    await registered.runs.wait_idle()

    state = seen[0]
    assert state.moral == MORAL_OPTIONS[2]
    assert state.age == 4


async def test_unexpected_text_in_age_state_shows_age_keyboard(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Текст вместо выбора возраста получает подсказку с клавиатурой."""

    registered, _ = _register(settings, repository, storage)
    event = FakeMessageCreated(FakeMessage())
    context = FakeContext(state=StoryFlow.waiting_age)

    await registered.other_message(event, context)

    assert event.message.sent_texts == [handlers_module.AGE_HINT]
    assert keyboard_labels(event.message.sent[-1].attachments) == [
        ["3 года"],
        ["4 года"],
        ["5 лет"],
        ["6 лет"],
    ]


async def test_unexpected_text_in_moral_state_shows_moral_keyboard(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Текст вместо выбора морали получает подсказку с моралью."""

    registered, _ = _register(settings, repository, storage)
    event = FakeMessageCreated(FakeMessage())
    context = FakeContext(state=StoryFlow.waiting_moral)

    await registered.other_message(event, context)

    assert event.message.sent_texts == [handlers_module.MORAL_HINT]
    assert keyboard_labels(event.message.sent[-1].attachments) == [
        [moral] for moral in MORAL_OPTIONS
    ]


async def test_fake_age_value_is_rejected_without_generation(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Значения вне клавиатуры не запускают генерацию."""

    pipeline = FakePipeline()
    registered, _ = _register(
        settings,
        repository,
        storage,
        pipeline=pipeline,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)

    event = FakeMessageCallback("age:9")
    await registered.callback(event, context)
    await registered.runs.wait_idle()

    assert event.sent_texts == [handlers_module.UNKNOWN_OPTION]
    assert context.state == StoryFlow.waiting_age
    assert pipeline.calls == []
    assert registered.runs.pending == 0


async def test_fake_moral_value_is_rejected_without_generation(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Мораль вне списка не запускает генерацию."""

    pipeline = FakePipeline()
    registered, _ = _register(
        settings,
        repository,
        storage,
        pipeline=pipeline,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)
    await registered.callback(FakeMessageCallback("age:4"), context)

    event = FakeMessageCallback("moral:99")
    await registered.callback(event, context)
    await registered.runs.wait_idle()

    assert event.sent_texts == [handlers_module.UNKNOWN_OPTION]
    assert context.state == StoryFlow.waiting_moral
    assert pipeline.calls == []


async def test_moral_without_photo_resets_dialog(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Мораль без принятого фото не запускает генерацию."""

    pipeline = FakePipeline()
    registered, _ = _register(settings, repository, storage, pipeline=pipeline)
    context = FakeContext(
        state=StoryFlow.waiting_moral, data={"age": 4, "tale_id": 1}
    )

    event = FakeMessageCallback("moral:0")
    await registered.callback(event, context)
    await registered.runs.wait_idle()

    assert event.sent_texts == [handlers_module.RESTART_HINT]
    assert pipeline.calls == []


async def test_quota_refusal_replaces_the_generation_promise(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Отказ по квоте приходит вместо «Запускаю сказку!».

    Проверка стоит в обработчике, до отправки обещания: иначе весь старый
    пользовательский состав видел сначала обещание, а следом отказ.
    """

    tale = await repository.create_tale(1)
    await repository.update_result(
        tale.id, status=GenerationStatus.SUCCESS, tale_text="Старая сказка"
    )

    sender = FakeSender()
    pipeline = FakePipeline()
    registered, _ = _register(
        settings,
        repository,
        storage,
        sender=sender,
        pipeline=pipeline,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)
    await registered.callback(FakeMessageCallback("age:4"), context)

    event = FakeMessageCallback("moral:0")
    await registered.callback(event, context)
    await registered.runs.wait_idle()

    assert event.sent_texts == [runner_module.FREE_LIMIT_REACHED]
    assert handlers_module.GENERATION_STARTED not in sender.sent_texts
    assert pipeline.calls == []
    # Запись о прогоне закрыта, а не осталась «в работе» навсегда.
    history = await repository.get_history(1)
    assert history[0].status is GenerationStatus.FAILED
    assert "квота" in (history[0].error or "")


async def test_quota_refusal_offers_subscription_button(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Отказ по квоте приходит с кнопкой подписки, и она работает."""

    tale = await repository.create_tale(1)
    await repository.update_result(
        tale.id, status=GenerationStatus.SUCCESS, tale_text="Старая сказка"
    )

    pipeline = FakePipeline()
    registered, _ = _register(
        settings,
        repository,
        storage,
        pipeline=pipeline,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)
    await registered.callback(FakeMessageCallback("age:4"), context)

    event = FakeMessageCallback("moral:0")
    await registered.callback(event, context)
    await registered.runs.wait_idle()

    assert keyboard_labels(event.sent[0].attachments) == [["Оформить подписку"]]

    stub = FakeMessageCallback("subscribe", message=FakeMessage())
    await registered.callback(stub, context)
    assert stub.sent_texts == [handlers_module.SUBSCRIPTION_STUB]


# --- 5.8: защита от повторного запуска ----------------------------------------


async def test_repeated_moral_click_answers_once_without_second_run(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Повторная доставка морали во время прогона не запускает второй прогон."""

    registered, _ = _register(
        settings,
        repository,
        storage,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)
    await registered.callback(FakeMessageCallback("age:4"), context)

    first = FakeMessageCallback("moral:0")
    await registered.callback(first, context)

    # Пока фоновая задача жива, реестр её видит.
    assert registered.runs.is_running(1) is True
    repeat = FakeMessageCallback("moral:0")
    await registered.callback(repeat, context)

    assert repeat.sent_texts == [handlers_module.ALREADY_RUNNING]
    await registered.runs.wait_idle()


async def test_text_message_during_generation_says_run_in_progress(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Сообщение во время генерации получает ответ, а не игнорирование."""

    import asyncio

    release = asyncio.Event()

    class _BlockedPipeline(FakePipeline):
        async def run(self, state):  # type: ignore[no-untyped-def]
            self.entered.set()
            await release.wait()
            return await super().run(state)

    pipeline = _BlockedPipeline()
    registered, _ = _register(
        settings,
        repository,
        storage,
        pipeline=pipeline,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)
    await registered.callback(FakeMessageCallback("age:4"), context)
    await registered.callback(FakeMessageCallback("moral:0"), context)

    # Диалог уже закончен, состояние ждёт фото — но прогон ещё идёт.
    assert context.state == StoryFlow.waiting_photo
    event = FakeMessageCreated(FakeMessage())
    await registered.other_message(event, context)

    assert event.message.sent_texts == [handlers_module.ALREADY_RUNNING]

    release.set()
    await registered.runs.wait_idle()


async def test_photo_during_generation_does_not_start_another_tale(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """Фото, присланное во время прогона, не создаёт вторую сказку."""

    import asyncio

    release = asyncio.Event()

    class _BlockedPipeline(FakePipeline):
        async def run(self, state):  # type: ignore[no-untyped-def]
            self.entered.set()
            await release.wait()
            return await super().run(state)

    registered, _ = _register(
        settings,
        repository,
        storage,
        pipeline=_BlockedPipeline(),
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)
    await registered.callback(FakeMessageCallback("age:4"), context)
    await registered.callback(FakeMessageCallback("moral:0"), context)

    event = FakeMessageCreated(
        FakeMessage(attachments=[make_image_attachment("http://max/again")])
    )
    await registered.photo(event, context)

    assert event.message.sent_texts == [handlers_module.ALREADY_RUNNING]
    # Второй прогон не заведён: платную работу не начинаем.
    assert len(await repository.get_history(1)) == 1

    release.set()
    await registered.runs.wait_idle()


async def test_after_generation_dialog_waits_photo_again(
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
) -> None:
    """После прогона состояние снова ждёт фото — «уже готовится» не зависает."""

    registered, _ = _register(
        settings,
        repository,
        storage,
        download_photo=_photo_download(make_photo_bytes(400, 400)),
    )
    context = FakeContext(state=StoryFlow.waiting_photo)
    await _prepare_dialog(registered, context)
    await registered.callback(FakeMessageCallback("age:4"), context)
    await registered.callback(FakeMessageCallback("moral:0"), context)
    await registered.runs.wait_idle()

    assert context.state == StoryFlow.waiting_photo
    assert registered.runs.is_running(1) is False

    event = FakeMessageCreated(FakeMessage())
    await registered.other_message(event, context)
    assert event.message.sent_texts == [handlers_module.NOT_A_PHOTO]
