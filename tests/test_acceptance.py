"""Приёмочные end-to-end тесты сценария бота (задачи 5.2–5.5, 7.5).

Полный путь на фейковых сервисах: /start → фото → возраст → мораль →
фоновая генерация → постраничная выдача → запись в БД. Живых вызовов нет.

Приёмка закрывает четыре вещи, которые продукт обещает пользователю:
квота (одна завершённая сказка), запись о прогоне в БД, постраничную выдачу
и частичный результат. Каждый сценарий идёт от события MAX до состояния
базы, а не через промежуточные хелперы.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.bot import handlers as handlers_module
from app.bot import runner as runner_module
from app.bot.handlers import register_handlers
from app.bot.keyboards import MORAL_OPTIONS
from app.bot.states import StoryFlow
from app.config import Settings
from app.db.engine import Database
from app.db.models import GenerationStatus, GenerationType
from app.db.repository import GenerationRepository
from app.services.storage import PhotoStorage
from app.story_pipeline.models import PageArt, PageText, StageMetrics, StoryState
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
from tests.factories import make_photo_bytes

TMP_ROOT = Path(".tmp/tests")

PHOTO = make_photo_bytes(400, 400)


def _story_state(
    run_dir: Path,
    *,
    pages: int = 3,
    illustrated: tuple[int, ...] | None = None,
    status: str = "ok",
    image_error: str | None = None,
) -> StoryState:
    """Состояние прогона с указанным числом страниц и нарисованных картинок.

    `illustrated` — номера страниц, для которых провайдер вернул картинку;
    по умолчанию нарисованы все. `image_error` кладёт причину в метрику этапа
    иллюстрирования, как это делает пайплайн при частичном результате.
    """

    numbers = range(1, pages + 1) if illustrated is None else illustrated
    images = []
    for number in numbers:
        image_path = run_dir / f"page_{number}.png"
        image_path.write_bytes(b"\x89PNG-page")
        images.append(
            PageArt(page_num=number, path=str(image_path), seed=1000 + number)
        )
    metrics = (
        [StageMetrics(stage="illustrate", duration_ms=5000, error=image_error)]
        if image_error is not None
        else []
    )
    return StoryState(
        theme="Сказка о любимой игрушке",
        age=5,
        moral=MORAL_OPTIONS[2],
        num_pages=pages,
        started_at="2026-01-01T00:00:00Z",  # type: ignore[arg-type]
        pages=[
            PageText(page_num=number, text=f"Текст страницы {number}.")
            for number in range(1, pages + 1)
        ],
        images=images,
        metrics=metrics,
        status=status,  # type: ignore[arg-type]
        output_dir=str(run_dir),
    )


def _page_messages(sender: FakeSender) -> list[object]:
    """Сообщения со страницами в порядке отправки."""

    return [
        item
        for item in sender.sent
        if isinstance(item.text, str) and item.text.startswith("Страница")
    ]


async def _run_dialog(
    database: Database,
    *,
    user_id: int = 1,
    state: StoryState,
) -> tuple[FakeSender, GenerationRepository, FakePipeline]:
    """Пройти весь диалог до конца фонового прогона.

    Общая часть всех приёмочных сценариев: /start не нужен, достаточно фото,
    возраста и морали. Возвращает отправитель, репозиторий и пайплайн, чтобы
    тест проверял то, что увидел бы пользователь и что оказалось в базе.
    """

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_bot_token="test-token",
    )
    repository = GenerationRepository(database)
    storage = PhotoStorage(TMP_ROOT / f"e2e-uploads-{uuid4().hex}")
    run_dir = TMP_ROOT / f"e2e-run-{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sender = FakeSender()
    pipeline = FakePipeline(result=state)

    async def download(url: str) -> bytes:
        return PHOTO

    services = make_services(
        settings=settings,
        repository=repository,
        storage=storage,
        sender=sender,
        vision=FakeVision(),
        pipeline=pipeline,
        download_photo=download,
    )
    registered = register_handlers(FakeDispatcher(), services)
    context = FakeContext()

    await registered.photo(
        FakeMessageCreated(
            FakeMessage(
                user_id=user_id,
                chat_id=user_id,
                attachments=[make_image_attachment("http://max/p")],
            )
        ),
        context,
    )
    await registered.callback(
        FakeMessageCallback("age:5", message=FakeMessage()),
        context,
    )
    await registered.callback(
        FakeMessageCallback("moral:2", message=FakeMessage()),
        context,
    )
    await registered.runs.wait_idle()

    return sender, repository, pipeline


async def test_end_to_end_acceptance(database: Database) -> None:
    """Полный сценарий: фото → возраст → мораль → постраничная сказка."""

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_bot_token="test-token",
    )

    repository = GenerationRepository(database)
    storage = PhotoStorage(TMP_ROOT / f"e2e-uploads-{uuid4().hex}")
    run_dir = TMP_ROOT / f"e2e-run-{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sender = FakeSender()
    vision = FakeVision()
    pipeline = FakePipeline(result=_story_state(run_dir))

    async def download(url: str) -> bytes:
        return make_photo_bytes(400, 400)

    services = make_services(
        settings=settings,
        repository=repository,
        storage=storage,
        sender=sender,
        vision=vision,
        pipeline=pipeline,
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

    # 2. Корректное фото → вопрос возраста.
    await registered.photo(
        FakeMessageCreated(
            FakeMessage(attachments=[make_image_attachment("http://max/p")])
        ),
        context,
    )
    assert context.state == StoryFlow.waiting_age

    # 3. Выбор возраста и морали.
    await registered.callback(
        FakeMessageCallback("age:5", message=FakeMessage()), context
    )
    assert context.state == StoryFlow.waiting_moral
    assert context.data["age"] == 5

    moral_callback = FakeMessageCallback("moral:2", message=FakeMessage())
    await registered.callback(moral_callback, context)
    assert moral_callback.sent_texts == [handlers_module.GENERATION_STARTED]

    # 4. Фоновая генерация отдала страницы по одной с картинками.
    await registered.runs.wait_idle()
    page_messages = _page_messages(sender)
    assert [item.text for item in page_messages] == [
        "Страница 1\nТекст страницы 1.",
        "Страница 2\nТекст страницы 2.",
        "Страница 3\nТекст страницы 3.",
    ]
    assert all(len(item.attachments or []) == 1 for item in page_messages)

    # 5. Пайплайн получил фото как референс, а прогон — свой каталог.
    call = pipeline.calls[0]
    assert call.reference_photo == make_photo_bytes(400, 400)
    assert call.settings.story_output_dir.startswith("output/1-")

    # 6. Запись в БД: успех, полный текст, путь к сохранённому фото.
    history = await repository.get_history(1)
    assert len(history) == 1
    record = history[0]
    assert record.type is GenerationType.TALE
    assert record.status is GenerationStatus.SUCCESS
    assert record.tale_text == (
        "Текст страницы 1. Текст страницы 2. Текст страницы 3."
    )
    assert Path(record.local_path).is_file()

    # 7. Ссылка на каталог прогона в чат не попала.
    assert not any(str(run_dir) in (item.text or "") for item in sender.sent)

    # 8. Бесплатная сказка выдана: квота израсходована.
    assert await repository.has_completed_story(1) is True


async def test_second_story_is_refused_by_quota(database: Database) -> None:
    """Вторая сказка того же пользователя не запускается: квота одна.

    Сценарий идёт от второго нажатия на мораль до сообщения в чат: пайплайн
    даже не собирается, потому что проверка квоты стоит первым шагом, до
    платного распознавания игрушки.
    """

    run_dir = TMP_ROOT / f"e2e-quota-first-{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sender, repository, pipeline = await _run_dialog(
        database, state=_story_state(run_dir)
    )
    assert await repository.has_completed_story(1) is True

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_bot_token="test-token",
    )
    storage = PhotoStorage(TMP_ROOT / f"e2e-quota-uploads-{uuid4().hex}")
    second_sender = FakeSender()
    second_pipeline = FakePipeline(error=AssertionError("пайплайн не собирался"))

    async def download(url: str) -> bytes:
        return PHOTO

    services = make_services(
        settings=settings,
        repository=repository,
        storage=storage,
        sender=second_sender,
        vision=FakeVision(),
        pipeline=second_pipeline,
        download_photo=download,
    )
    registered = register_handlers(FakeDispatcher(), services)
    context = FakeContext()

    await registered.photo(
        FakeMessageCreated(
            FakeMessage(attachments=[make_image_attachment("http://max/p")])
        ),
        context,
    )
    await registered.callback(
        FakeMessageCallback("age:5", message=FakeMessage()), context
    )
    await registered.callback(
        moral_event := FakeMessageCallback("moral:2", message=FakeMessage()),
        context,
    )
    await registered.runs.wait_idle()

    # Пайплайн не запускался: платный вызов не оплачен впустую.
    assert second_pipeline.calls == []
    # Отказ пришёл на нажатие вместо обещания «Запускаю сказку!».
    assert moral_event.sent_texts == [runner_module.FREE_LIMIT_REACHED]
    assert handlers_module.GENERATION_STARTED not in second_sender.sent_texts
    # Второй прогон закрыт как не начавшийся, а не остался «в работе».
    history = await repository.get_history(1)
    assert [record.status for record in history] == [
        GenerationStatus.FAILED,
        GenerationStatus.SUCCESS,
    ]
    assert "квота" in (history[0].error or "")


async def test_partial_result_is_delivered_and_quota_is_kept(
    database: Database,
) -> None:
    """Частичный результат: текст отдан, квота не израсходована.

    Родитель получил не всю книжку, поэтому бесплатная сказка ему не
    засчитывается: иначе второй не оплаченный по сути прогон был бы
    последним. Проверяются все три следствия: доставленные страницы, честное
    объяснение с номерами и запись в БД со статусом `FAILED`.
    """

    run_dir = TMP_ROOT / f"e2e-partial-{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sender, repository, _ = await _run_dialog(
        database,
        state=_story_state(
            run_dir,
            pages=3,
            illustrated=(1, 3),
            status="needs_review",
            image_error="routerai: 503",
        ),
    )

    # Страницы приходят все три, но картинка — только у первой и третьей.
    page_messages = _page_messages(sender)
    assert [item.text for item in page_messages][0] == (
        "Страница 1\nТекст страницы 1."
    )
    assert [item.text for item in page_messages][1].startswith("Страница 2")
    assert runner_module.PAGE_IMAGE_SKIPPED in page_messages[1].text
    assert [item.text for item in page_messages][2].startswith("Страница 3")
    assert len(page_messages[0].attachments or []) == 1
    assert not page_messages[1].attachments
    assert len(page_messages[2].attachments or []) == 1

    # Пользователю сказано ровно то, чего не хватило, и предложен один повтор.
    texts = [text for text in sender.sent_texts if isinstance(text, str)]
    summary = next(
        text
        for text in texts
        if text.startswith(runner_module.PARTIAL_TITLE)
    )
    assert "2" in summary
    assert runner_module.PARTIAL_REASON in summary
    assert runner_module.PARTIAL_REDO in texts
    # Итогового «сказка готова» при частичном результате быть не должно.
    assert runner_module.STORY_READY not in texts

    # Запись закрыта как неуспешная, но текст и число страниц сохранены.
    record = (await repository.get_history(1))[0]
    assert record.status is GenerationStatus.FAILED
    assert record.pages_count == 3
    assert record.tale_text == (
        "Текст страницы 1. Текст страницы 2. Текст страницы 3."
    )
    assert "routerai" in (record.error or "")
    assert await repository.has_completed_story(1) is False
