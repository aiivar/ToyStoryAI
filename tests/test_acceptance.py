"""Приёмочный end-to-end тест сценария бота (задача 5.2–5.5).

Полный путь на фейковых сервисах: /start → фото → возраст → мораль →
фоновая генерация → постраничная выдача → запись в БД. Живых вызовов нет.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.bot import handlers as handlers_module
from app.bot.handlers import register_handlers
from app.bot.keyboards import MORAL_OPTIONS
from app.bot.states import StoryFlow
from app.config import Settings
from app.db.engine import Database
from app.db.models import GenerationStatus, GenerationType
from app.db.repository import GenerationRepository
from app.services.storage import PhotoStorage
from app.story_pipeline.models import PageArt, PageText, StoryState
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


def _story_state(run_dir: Path, *, pages: int = 3) -> StoryState:
    """Состояние прогона с тремя страницами и нарисованными картинками."""

    images = []
    for number in range(1, pages + 1):
        image_path = run_dir / f"page_{number}.png"
        image_path.write_bytes(b"\x89PNG-page")
        images.append(
            PageArt(page_num=number, path=str(image_path), seed=1000 + number)
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
        output_dir=str(run_dir),
    )


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
    page_messages = [
        item
        for item in sender.sent
        if isinstance(item.text, str) and item.text.startswith("Страница")
    ]
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
    assert not any(
        str(run_dir) in (item.text or "") for item in sender.sent
    )
