"""Тесты фоновой генерации: отправитель, реестр задач, вход пайплайна.

Задачи 5.3, 5.4, 5.7: отправитель привязан к боту и работает из фоновой
задачи, обработчик возвращается до конца генерации, а в пайплайн уходят
детерминированная тема, описание игрушки, байты фото как референс и число
страниц из настроек.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from maxapi.enums.upload_type import UploadType
from maxapi.types.input_media import InputMediaBuffer

from app.bot import runner as runner_module
from app.bot.messaging import ChatSender
from app.bot.runner import PAGE_TITLE
from app.bot.runner import RunRequest, StoryRunner
from app.db.models import GenerationStatus
from app.services.toy_vision import ToyDescription
from app.story_pipeline.errors import StoryPipelineError
from app.story_pipeline.models import (
    PageArt,
    PageText,
    StageMetrics,
    StoryState,
)
from app.story_pipeline.settings import StorySettings
from tests.bot_harness import keyboard_labels
from tests.factories import make_jpeg, make_webp

TMP_ROOT = Path(".tmp/tests")

DESCRIPTION = ToyDescription(
    kind="плюшевый медвежонок",
    color="рыжий",
    material="мягкая ткань",
    features="кисточка на голове, красный шарф",
    short="рыжий плюшевый медвежонок в красном шарфе",
)

MORAL = "помогать тем, кому трудно"

# Этапы, о которых пайплайн сообщает через `on_stage`.
STAGE_NAMES = ("plan", "screenwrite", "quality", "illustrate")


class FakeBot:
    """Заглушка `Bot.send_message`, записывающая исходящие сообщения.

    `fail_attachments=True` имитирует отказ загрузки файла в MAX: такое
    сообщение не доставляется вовсе, поэтому проверка «текст страницы не
    потерян» должна видеть повторную отправку без вложения.
    """

    def __init__(self, *, fail_attachments: bool = False) -> None:
        self.sent: list[SimpleNamespace] = []
        self.fail_attachments = fail_attachments

    async def send_message(
        self,
        chat_id: int | None = None,
        user_id: int | None = None,
        text: str | None = None,
        attachments: list[Any] | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        if attachments and self.fail_attachments:
            raise RuntimeError("загрузка файла в MAX отклонена")
        self.sent.append(
            SimpleNamespace(chat_id=chat_id, text=text, attachments=attachments)
        )
        return SimpleNamespace(message_id="m1")

    @property
    def sent_texts(self) -> list[str | None]:
        return [item.text for item in self.sent]

    @property
    def sent_chat_ids(self) -> list[int | None]:
        return [item.chat_id for item in self.sent]


class FakeVision:
    """Заглушка `ToyVisionService`, отдающая заранее заданное описание."""

    def __init__(
        self,
        description: ToyDescription | None = DESCRIPTION,
        error: Exception | None = None,
    ) -> None:
        self._description = description
        self._error = error
        self.photos: list[bytes] = []
        self.closed = 0

    async def describe(self, photo: bytes) -> ToyDescription:
        self.photos.append(photo)
        if self._error is not None:
            raise self._error
        if self._description is None:  # pragma: no cover - опечатка в тесте
            raise AssertionError("Описание игрушки не задано")
        return self._description

    async def aclose(self) -> None:
        self.closed += 1


class RecordingPipeline:
    """Заглушка `StoryPipeline`: пишет вход прогона и повторяет этапы."""

    def __init__(
        self,
        result: StoryState | None = None,
        error: Exception | None = None,
        stage_names: tuple[str, ...] = STAGE_NAMES,
    ) -> None:
        self._result = result
        self._error = error
        self._stage_names = stage_names
        self._on_stage: Any = None
        self.closed = 0
        # Аргументы фабрики: настройки, фото-референс, колбэк прогресса.
        self.calls: list[SimpleNamespace] = []
        # Этапы, о которых пайплайн сообщил через `on_stage`.
        self.stages: list[str] = []
        # Момент входа в прогон: позволяет доказать, что обработчик ушёл
        # раньше генерации.
        self.entered = asyncio.Event()

    def bind(self, kwargs: dict[str, Any]) -> None:
        """Запомнить аргументы, с которыми собрана заглушка пайплайна."""

        self.calls.append(SimpleNamespace(**kwargs))
        self._on_stage = kwargs.get("on_stage")

    async def run(self, state: StoryState) -> StoryState:
        self.entered.set()
        if self._on_stage is not None:
            for name in self._stage_names:
                self.stages.append(name)
                await self._on_stage(name)
        if self._error is not None:
            raise self._error
        if self._result is None:  # pragma: no cover - опечатка в тесте
            raise AssertionError("Результат прогона не задан")
        return self._result

    async def aclose(self) -> None:
        self.closed += 1


class BlockingPipeline(RecordingPipeline):
    """Пайплайн, который ждёт разрешения: удобно для проверки реестра задач."""

    def __init__(self, *, release: asyncio.Event, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._release = release

    async def run(self, state: StoryState) -> StoryState:
        self.entered.set()
        await self._release.wait()
        return await super().run(state)


class FakeRepository:
    """Заглушка репозитория: помнит вызовы и отвечает про квоту."""

    def __init__(self, *, story_used: bool = False) -> None:
        self.updates: list[dict[str, Any]] = []
        self.story_used = story_used
        self.quota_checks: list[int] = []

    async def update_result(
        self,
        generation_id: int,
        **kwargs: Any,
    ) -> None:
        self.updates.append({"generation_id": generation_id, **kwargs})

    async def has_completed_story(self, user_id: int) -> bool:
        self.quota_checks.append(user_id)
        return self.story_used

    @property
    def statuses(self) -> list[Any]:
        return [update["status"] for update in self.updates]


@dataclass
class RunnerBundle:
    """Собранные фейки и раннер под руку."""

    runner: StoryRunner
    bot: FakeBot
    repository: FakeRepository
    pipeline: RecordingPipeline


def make_runner(
    *,
    pipeline: RecordingPipeline | None = None,
    vision: FakeVision | None = None,
    settings: StorySettings | None = None,
    bot: FakeBot | None = None,
    repository: FakeRepository | None = None,
) -> RunnerBundle:
    """Собрать раннер с фейковыми ботом, репозиторием и пайплайном."""

    bot = bot or FakeBot()
    pipeline = pipeline or RecordingPipeline()
    repository = repository or FakeRepository()

    def factory(**kwargs: Any) -> RecordingPipeline:
        pipeline.bind(kwargs)
        return pipeline

    runner = StoryRunner(
        repository=repository,
        vision=vision or FakeVision(),
        make_pipeline=factory,
        story_settings=(
            settings
            if settings is not None
            else StorySettings(_env_file=None)  # type: ignore[call-arg]
        ),
        sender=ChatSender(bot),
    )
    return RunnerBundle(
        runner=runner, bot=bot, repository=repository, pipeline=pipeline
    )


def make_request(
    *,
    tale_id: int = 1,
    user_id: int = 1,
    chat_id: int = 1,
    age: int = 4,
    moral: str = MORAL,
    photo: bytes = b"\x89PNG-photo",
    photo_path: str | None = None,
) -> RunRequest:
    """Запрос на фоновую генерацию."""

    return RunRequest(
        user_id=user_id,
        chat_id=chat_id,
        tale_id=tale_id,
        photo_path=photo_path or str(TMP_ROOT / f"photo-{uuid4().hex}.png"),
        photo=photo,
        age=age,
        moral=moral,
    )


def made_state(
    tmp_dir: Path,
    *,
    status: str = "ok",
    pages: int = 1,
    illustrated: tuple[int, ...] | None = None,
    illustrated_set: tuple[int, ...] | None = None,
    image_bytes: bytes = b"\x89PNG-page",
    extension: str = ".png",
) -> StoryState:
    """Состояние прогона с заданными страницами и нарисованными картинками.

    `extension` повторяет соглашение пайплайна: имя файла страницы строится
    по типу, который вернул провайдер, поэтому JPEG-файл лежит под `.jpg`.

    Различаются два параметра, потому что различаются и две вещи:

    - `illustrated` — у каких страниц картинка фактически есть;
    - `illustrated_set` — набор страниц, которым картинка заказывалась.

    Обычно они совпадают, и расходятся как раз в сценарии сбоя на
    иллюстрируемой странице. `illustrated_set=None` оставляет поле пустым, то
    есть «набор не задан» и иллюстрируются все страницы: так проверяется
    прежнее поведение без ограничения.
    """

    illustrated = (
        tuple(range(1, pages + 1)) if illustrated is None else illustrated
    )
    page_texts = [
        PageText(page_num=number, text=f"Текст страницы {number}.")
        for number in range(1, pages + 1)
    ]
    images: list[PageArt] = []
    for number in illustrated:
        image_path = tmp_dir / f"page_{number}{extension}"
        image_path.write_bytes(image_bytes)
        images.append(
            PageArt(page_num=number, path=str(image_path), seed=1000 + number)
        )
    return StoryState(
        theme="тема",
        age=4,
        moral=MORAL,
        num_pages=pages,
        illustrated_pages=list(illustrated_set or []),
        started_at="2026-01-01T00:00:00Z",  # type: ignore[arg-type]
        pages=page_texts,
        images=images,
        status=status,  # type: ignore[arg-type]
    )


# --- 5.7: отправитель, привязанный к боту ------------------------------------


async def test_sender_sends_through_bot_by_chat_id() -> None:
    """Отправитель работает вне события: только `Bot` и явный `chat_id`."""

    bot = FakeBot()
    sender = ChatSender(bot)

    await sender.send(42, text="Страница 1", attachments=None)

    assert len(bot.sent) == 1
    assert bot.sent[0].chat_id == 42
    assert bot.sent[0].text == "Страница 1"


# --- 5.4: генерация в фоне ----------------------------------------------------


async def test_start_returns_before_generation_finishes(tmp_path: Path) -> None:
    """`start` возвращает управление, а генерация ещё не началась."""

    bundle = make_runner(pipeline=RecordingPipeline(result=made_state(tmp_path)))

    assert bundle.runner.start(make_request()) is True

    assert bundle.pipeline.entered.is_set() is False
    await bundle.runner.wait_idle()
    assert bundle.pipeline.entered.is_set() is True
    assert bundle.pipeline.stages == list(STAGE_NAMES)


async def test_second_start_for_same_user_is_rejected(tmp_path: Path) -> None:
    """Пока идёт прогон пользователя, второй не запускается."""

    release = asyncio.Event()
    bundle = make_runner(
        pipeline=BlockingPipeline(
            result=made_state(tmp_path), release=release
        )
    )

    assert bundle.runner.start(make_request()) is True
    assert bundle.runner.start(make_request(tale_id=2)) is False
    assert bundle.runner.is_running(1) is True

    release.set()
    await bundle.runner.wait_idle()
    assert bundle.runner.is_running(1) is False


async def test_progress_messages_sent_once_per_kind(tmp_path: Path) -> None:
    """О прогрессе сообщают один раз: подготовка и рисование страниц."""

    bundle = make_runner(pipeline=RecordingPipeline(result=made_state(tmp_path)))

    await bundle.runner.run(make_request())

    assert bundle.bot.sent_texts.count(runner_module.STORY_PROGRESS) == 1
    assert bundle.bot.sent_texts.count(runner_module.ILLUSTRATION_PROGRESS) == 1
    assert bundle.bot.sent_texts.index(
        runner_module.STORY_PROGRESS
    ) < bundle.bot.sent_texts.index(runner_module.ILLUSTRATION_PROGRESS)


async def test_task_stays_in_registry_until_done(tmp_path: Path) -> None:
    """Задача хранится в реестре до завершения, иначе её соберёт GC."""

    bundle = make_runner(pipeline=RecordingPipeline(result=made_state(tmp_path)))

    bundle.runner.start(make_request())
    assert bundle.runner.pending == 1

    await bundle.runner.wait_idle()
    assert bundle.runner.pending == 0


# --- 5.3: вход пайплайна ------------------------------------------------------


async def test_pipeline_receives_photo_reference_and_run_directory(
    tmp_path: Path,
) -> None:
    """В пайплайн уходят байты фото и уникальный каталог прогона."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path)),
        settings=StorySettings(_env_file=None),  # type: ignore[call-arg]
    )
    photo = b"\x89PNG-photo-bytes"

    await bundle.runner.run(make_request(photo=photo))

    call = bundle.pipeline.calls[0]
    assert call.reference_photo == photo
    assert call.settings.story_output_dir.startswith("output/1-")


async def test_two_runs_of_one_user_get_different_directories(
    tmp_path: Path,
) -> None:
    """Каталоги двух прогонов одного пользователя не совпадают."""

    bundle = make_runner(pipeline=RecordingPipeline(result=made_state(tmp_path)))

    await bundle.runner.run(make_request(tale_id=1))
    await bundle.runner.run(make_request(tale_id=2))

    first, second = (call.settings.story_output_dir for call in bundle.pipeline.calls)
    assert first != second


async def test_state_carries_deterministic_theme_and_toy_description(
    tmp_path: Path,
) -> None:
    """Тема и описание игрушки собираются из распознавания, морали и возраста."""

    seen: list[StoryState] = []

    class _CapturingPipeline(RecordingPipeline):
        async def run(self, state: StoryState) -> StoryState:
            seen.append(state)
            return await super().run(state)

    bundle = make_runner(pipeline=_CapturingPipeline(result=made_state(tmp_path)))

    await bundle.runner.run(make_request(age=6, moral=MORAL))

    state = seen[0]
    assert state.age == 6
    assert state.moral == MORAL
    assert DESCRIPTION.short in state.theme
    assert MORAL in state.theme
    assert state.toy_description is not None
    assert DESCRIPTION.features in state.toy_description
    assert state.num_pages == bundle.pipeline.calls[0].settings.story_default_pages


# --- Квота: вторая платная сказка не запускается ------------------------------


async def test_used_story_blocks_generation_before_any_paid_work(
    tmp_path: Path,
) -> None:
    """Израсходованная квота останавливает прогон до распознавания и пайплайна."""

    vision = FakeVision()
    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path)),
        vision=vision,
        repository=FakeRepository(story_used=True),
    )

    await bundle.runner.run(make_request())

    assert bundle.repository.quota_checks == [1]
    assert vision.photos == []  # платное распознавание не вызывалось
    assert bundle.pipeline.calls == []  # платный пайплайн не собирался
    assert bundle.bot.sent_texts == [runner_module.FREE_LIMIT_REACHED]
    assert keyboard_labels(bundle.bot.sent[0].attachments) == [
        ["Оформить подписку"]
    ]
    # Запись о прогоне закрыта, а не осталась «в работе» навсегда.
    assert bundle.repository.updates[-1]["status"] is GenerationStatus.FAILED
    assert "квота" in str(bundle.repository.updates[-1]["error"])


async def test_free_story_is_allowed_for_new_user(tmp_path: Path) -> None:
    """Без завершённых сказок прогон идёт как обычно."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path)),
        repository=FakeRepository(story_used=False),
    )

    await bundle.runner.run(make_request())

    assert bundle.repository.quota_checks == [1]
    assert bundle.pipeline.calls
    assert runner_module.FREE_LIMIT_REACHED not in [
        text for text in bundle.bot.sent_texts
    ]


def test_quota_text_talks_about_story_only() -> None:
    """Текст исчерпанной квоты — про сказку и без упоминания видео."""

    text = runner_module.FREE_LIMIT_REACHED.lower()
    assert "сказк" in text
    assert "видео" not in text
    assert "подписк" in text


# --- 5.5: постраничная выдача ------------------------------------------------


def page_messages(bot: FakeBot) -> list[SimpleNamespace]:
    """Сообщения со страницами: заголовок «Страница N» + текст."""

    return [
        item
        for item in bot.sent
        if isinstance(item.text, str) and item.text.startswith(PAGE_TITLE)
    ]


async def test_pages_sent_in_order_with_one_attachment_each(
    tmp_path: Path,
) -> None:
    """На каждую страницу одно сообщение: «Страница N», текст и картинка."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path, pages=3))
    )

    await bundle.runner.run(make_request())

    pages = page_messages(bundle.bot)
    assert [item.text for item in pages] == [
        "Страница 1\nТекст страницы 1.",
        "Страница 2\nТекст страницы 2.",
        "Страница 3\nТекст страницы 3.",
    ]
    assert len(pages) == 3
    assert [
        item.attachments[0].filename for item in pages
    ] == ["page_1.png", "page_2.png", "page_3.png"]


async def test_attachment_bytes_and_type_match_page_file(
    tmp_path: Path,
) -> None:
    """Во вложение уходят байты файла страницы и тип по содержимому."""

    bundle = make_runner(
        pipeline=RecordingPipeline(
            result=made_state(
                tmp_path,
                pages=1,
                image_bytes=make_jpeg(400, 300),
                extension=".jpg",
            )
        )
    )

    await bundle.runner.run(make_request())

    attachment = page_messages(bundle.bot)[0].attachments[0]
    assert isinstance(attachment, InputMediaBuffer)
    assert attachment.buffer == make_jpeg(400, 300)
    assert attachment.type == UploadType.IMAGE
    assert attachment.filename == "page_1.jpg"


async def test_mime_follows_real_file_type_not_guessed(
    tmp_path: Path,
) -> None:
    """Имя и тип вложения берутся из файла прогона, а не из догадки."""

    state = made_state(
        tmp_path, pages=1, image_bytes=make_webp(400, 300), extension=".webp"
    )

    bundle = make_runner(pipeline=RecordingPipeline(result=state))
    await bundle.runner.run(make_request())

    attachment = page_messages(bundle.bot)[0].attachments[0]
    assert attachment.filename == "page_1.webp"
    assert attachment.type == UploadType.IMAGE


async def test_all_messages_carry_one_page_only(tmp_path: Path) -> None:
    """Одно вложение на сообщение: лишних картинок в сообщении быть не может."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path, pages=4))
    )

    await bundle.runner.run(make_request())

    for item in bundle.bot.sent:
        assert len(item.attachments or []) <= 1


async def test_success_record_contains_full_text_and_success_status(
    tmp_path: Path,
) -> None:
    """В записи о прогоне есть статус успеха и полный текст сказки."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path, pages=2))
    )

    await bundle.runner.run(make_request(tale_id=7))

    update = bundle.repository.updates[-1]
    assert update["generation_id"] == 7
    assert update["status"] is GenerationStatus.SUCCESS
    assert update["tale_text"] == "Текст страницы 1. Текст страницы 2."


# --- 6.4: содержимое записи о прогоне ----------------------------------------


async def test_successful_run_records_pages_count_and_output_dir(
    tmp_path: Path,
) -> None:
    """Успешный прогон пишет число страниц и каталог прогона (D19)."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path, pages=3))
    )

    await bundle.runner.run(make_request(tale_id=8))

    update = bundle.repository.updates[-1]
    assert update["generation_id"] == 8
    assert update["pages_count"] == 3
    assert update["output_dir"].startswith("output/1-")
    # Каталог прогона — диагностика, ссылку на него в чат не отдаём.
    texts = [text for text in bundle.bot.sent_texts if text]
    assert not any(update["output_dir"] in text for text in texts)


async def test_partial_run_records_pages_and_failed_status(
    tmp_path: Path,
) -> None:
    """Частичный прогон: страницы и каталог записаны, статус FAILED."""

    bundle = make_runner(
        pipeline=RecordingPipeline(
            result=made_state(
                tmp_path,
                pages=2,
                illustrated=(1,),
                status="needs_review",
            )
        )
    )

    await bundle.runner.run(make_request(tale_id=9))

    update = bundle.repository.updates[-1]
    assert update["status"] is GenerationStatus.FAILED
    assert update["tale_text"] == "Текст страницы 1. Текст страницы 2."
    assert update["pages_count"] == 2
    assert str(update["output_dir"]).startswith("output/1-")


async def test_recorded_dir_is_the_one_given_to_pipeline(
    tmp_path: Path,
) -> None:
    """В записи тот же каталог, который раннер отдал пайплайну (D16)."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path, pages=1))
    )

    await bundle.runner.run(make_request())

    sent_dir = bundle.pipeline.calls[0].settings.story_output_dir
    assert bundle.repository.updates[-1]["output_dir"] == sent_dir


async def test_crashed_run_still_records_run_dir(tmp_path: Path) -> None:
    """Обрыв после сборки настроек оставляет каталог прогона в записи."""

    bundle = make_runner(
        pipeline=RecordingPipeline(error=RuntimeError("провайдер упал"))
    )

    await bundle.runner.run(make_request())

    update = bundle.repository.updates[-1]
    assert update["status"] is GenerationStatus.FAILED
    assert update["output_dir"] == (
        bundle.pipeline.calls[0].settings.story_output_dir
    )


# --- 5.6: отказы --------------------------------------------------------------


async def test_vision_failure_explains_and_records_failure(
    tmp_path: Path,
) -> None:
    """Ошибка распознавания: понятный текст, статус в БД, без трассировки."""

    from app.services.toy_vision import PermanentToyVisionError

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path)),
        vision=FakeVision(
            error=PermanentToyVisionError("модель не видит картинки")
        ),
    )

    await bundle.runner.run(make_request())

    assert bundle.bot.sent_texts == [runner_module.VISION_FAILED]
    assert "Traceback" not in (bundle.bot.sent_texts[0] or "")
    update = bundle.repository.updates[-1]
    assert update["status"] is GenerationStatus.FAILED
    assert "модель не видит картинки" in str(update["error"])
    assert bundle.pipeline.calls == []  # до пайплайна не дошли


async def test_pipeline_failure_explains_and_records_failure(
    tmp_path: Path,
) -> None:
    """Сбой пайплайна: понятный текст, провайдеры закрыты, статус в БД."""

    bundle = make_runner(
        pipeline=RecordingPipeline(
            error=StoryPipelineError("платный вызов не прошёл")
        )
    )

    await bundle.runner.run(make_request(tale_id=3))

    assert bundle.bot.sent_texts[-1] == runner_module.GENERATION_FAILED
    assert "Traceback" not in (bundle.bot.sent_texts[-1] or "")
    update = bundle.repository.updates[-1]
    assert update["generation_id"] == 3
    assert update["status"] is GenerationStatus.FAILED
    assert bundle.pipeline.closed == 1


async def test_unexpected_error_in_task_is_recorded(tmp_path: Path) -> None:
    """Обрыв фоновой задачи не остаётся незамеченным: статус пишется в БД."""

    class _ExplodingPipeline(RecordingPipeline):
        async def run(self, state: StoryState) -> StoryState:
            self.entered.set()
            raise RuntimeError("непредвиденный обрыв")

    bundle = make_runner(pipeline=_ExplodingPipeline())
    request = make_request(tale_id=9)

    assert bundle.runner.start(request) is True
    await bundle.runner.wait_idle()

    update = bundle.repository.updates[-1]
    assert update["generation_id"] == 9
    assert update["status"] is GenerationStatus.FAILED
    assert "непредвиденный обрыв" in str(update["error"])
    assert bundle.runner.pending == 0


async def test_failing_pipeline_factory_explains_to_user(tmp_path: Path) -> None:
    """Сбой сборки пайплайна не оставляет пользователя в тишине.

    Сборка провайдеров (клиент LLM, провайдер картинок, валидация настроек)
    стоит до `pipeline.run()`, поэтому раньше её исключение уходило мимо
    сообщения пользователю: тот видел «Запускаю сказку!» и больше ничего.
    """

    bot = FakeBot()
    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path)), bot=bot
    )

    def broken_factory(**kwargs: Any) -> RecordingPipeline:
        raise ValueError("STORY_IMAGE_ASPECT_RATIO: неподдерживаемое значение")

    bundle.runner._make_pipeline = broken_factory  # noqa: SLF001 - подмена фейка

    await bundle.runner.run(make_request(tale_id=5))

    assert bundle.bot.sent_texts == [runner_module.GENERATION_FAILED]
    assert "Traceback" not in (bundle.bot.sent_texts[0] or "")
    assert bundle.repository.updates[-1]["generation_id"] == 5
    assert bundle.repository.updates[-1]["status"] is GenerationStatus.FAILED


async def test_failing_factory_in_background_task_sends_message(
    tmp_path: Path,
) -> None:
    """То же самое через фоновую задачу: сообщение уходит, статус пишется."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path))
    )

    def broken_factory(**kwargs: Any) -> RecordingPipeline:
        raise RuntimeError("провайдер не собирается")

    bundle.runner._make_pipeline = broken_factory  # noqa: SLF001 - подмена фейка

    assert bundle.runner.start(make_request(tale_id=6)) is True
    await bundle.runner.wait_idle()

    assert bundle.bot.sent_texts == [runner_module.GENERATION_FAILED]
    assert bundle.repository.updates[-1]["generation_id"] == 6
    assert "провайдер не собирается" in str(
        bundle.repository.updates[-1]["error"]
    )


async def test_missing_page_file_keeps_page_text(tmp_path: Path) -> None:
    """Пропавший файл страницы: текст отправляется, ссылка на каталог — нет."""

    state = made_state(tmp_path, pages=1)
    Path(state.images[0].path).unlink()
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    texts = [text for text in bundle.bot.sent_texts if text is not None]
    assert any("Страница 1" in text for text in texts)
    assert runner_module.PAGE_WITHOUT_IMAGE in texts[-2]
    assert not any("output/" in text or ".tmp" in text for text in texts)
    # Запись о прогоне при этом остаётся успешной: текст-то отправлен.
    assert bundle.repository.updates[-1]["status"] is GenerationStatus.SUCCESS


async def test_failed_attachment_upload_resends_page_text(
    tmp_path: Path,
) -> None:
    """Отказ загрузки вложения не отнимает у пользователя текст страницы."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path, pages=1)),
        bot=FakeBot(fail_attachments=True),
    )

    await bundle.runner.run(make_request())

    texts = [text for text in bundle.bot.sent_texts if text is not None]
    page_texts = [text for text in texts if "Страница 1" in text]
    # Отправка с вложением упала целиком и не была доставлена, поэтому текст
    # страницы уходит один раз — уже с объяснением, без картинки.
    assert page_texts == [
        f"Страница 1\nТекст страницы 1.\n\n{runner_module.PAGE_WITHOUT_IMAGE}"
    ]


# --- 5.9: частичный результат -------------------------------------------------


async def test_partial_result_delivers_ready_pages_and_explains_missing(
    tmp_path: Path,
) -> None:
    """Частичный прогон: готовые страницы отданы, пропуск объяснён."""

    state = made_state(
        tmp_path,
        status="needs_review",
        pages=3,
        illustrated=(1, 2),
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    delivered = page_messages(bundle.bot)
    assert [item.text for item in delivered] == [
        "Страница 1\nТекст страницы 1.",
        "Страница 2\nТекст страницы 2.",
        "Страница 3\nТекст страницы 3.\n\n" + runner_module.PAGE_IMAGE_SKIPPED,
    ]
    assert any(
        runner_module.PARTIAL_TITLE in (text or "")
        for text in bundle.bot.sent_texts
    )


async def test_partial_result_names_missing_pages_and_reason(
    tmp_path: Path,
) -> None:
    """Пользователь узнаёт, каких именно страниц нет и почему."""

    state = made_state(
        tmp_path,
        status="needs_review",
        pages=4,
        illustrated=(1, 2),
    )
    state.metrics.append(
        StageMetrics(
            stage="illustrate",
            duration_ms=10.0,
            error=(
                "illustration: page 3 failed after retries: "
                "TransientImageGenerationError "
                "(HTTP 429) https://api.routerai.ru/api/v1/images"
            ),
        )
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    summary = next(
        text
        for text in bundle.bot.sent_texts
        if text and text.startswith(runner_module.PARTIAL_TITLE)
    )
    assert "3, 4" in summary
    assert runner_module.PARTIAL_REASON in summary
    # Техническая причина провайдера в чат не попадает: она английская и
    # может содержать URL.
    assert "TransientImageGenerationError" not in summary
    assert "routerai.ru" not in summary
    assert "Traceback" not in summary
    # В запись о прогоне техническая причина, наоборот, нужна.
    update = bundle.repository.updates[-1]
    assert "TransientImageGenerationError" in str(update["error"])


async def test_partial_result_reason_has_russian_fallback(
    tmp_path: Path,
) -> None:
    """Без метрик с ошибкой показывается понятная причина, а не пустота."""

    state = made_state(
        tmp_path, status="needs_review", pages=1, illustrated=()
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    summary = next(
        text
        for text in bundle.bot.sent_texts
        if text and text.startswith(runner_module.PARTIAL_TITLE)
    )
    assert runner_module.PARTIAL_REASON in summary
    assert "None" not in summary


async def test_reviewed_text_with_all_pictures_is_not_called_partial(
    tmp_path: Path,
) -> None:
    """Все картинки на месте: книжка полная, а не «частичный результат».

    Статус `needs_review` ставится и при исчерпании бюджета ревизий, когда
    иллюстрирование идёт штатно. Раньше такая книжка доставлялась целиком,
    а следом приходило «без картинок остались страницы » с пустым списком и
    выдуманной причиной про отказ сервиса картинок.
    """

    state = made_state(tmp_path, status="needs_review", pages=2)
    state.metrics.append(
        StageMetrics(
            stage="quality",
            duration_ms=10.0,
            error="judge: passed=false after 2 revisions",
        )
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request(tale_id=11))

    texts = [text for text in bundle.bot.sent_texts if text]
    assert not any(text.startswith(runner_module.PARTIAL_TITLE) for text in texts)
    summary = next(
        text for text in texts if text.startswith(runner_module.REVIEW_TITLE)
    )
    # Причина в чате — про текст, а не про отказ сервиса картинок.
    assert runner_module.PARTIAL_REASON not in summary
    assert "отрисовать" not in summary
    assert "не полной" not in summary
    # Техническая причина сорвавшегося этапа — в запись и лог, но не в чат.
    assert "judge: passed=false" not in summary
    update = bundle.repository.updates[-1]
    assert "judge: passed=false" in str(update["error"])
    # Страницы с картинками доставлены все.
    delivered = page_messages(bundle.bot)
    assert [item.text for item in delivered] == [
        "Страница 1\nТекст страницы 1.",
        "Страница 2\nТекст страницы 2.",
    ]
    assert all(item.attachments for item in delivered)


async def test_reviewed_text_book_consumes_quota(tmp_path: Path) -> None:
    """Книга отдана целиком — она засчитывается как завершённая сказка.

    Квота привязана к статусу `SUCCESS` (`has_completed_story`). Книжка со
    всеми страницами и картинками — это продукт, который пользователь получил,
    поэтому и статус, и квота обязаны говорить одно и то же.
    """

    bundle = make_runner(
        pipeline=RecordingPipeline(
            result=made_state(tmp_path, status="needs_review", pages=2)
        )
    )

    await bundle.runner.run(make_request(tale_id=12))

    update = bundle.repository.updates[-1]
    assert update["status"] is GenerationStatus.SUCCESS
    assert update["pages_count"] == 2
    assert update["tale_text"] == "Текст страницы 1. Текст страницы 2."


async def test_reviewed_text_book_offers_no_redo(tmp_path: Path) -> None:
    """Переделывать нечего: книжка отдана, а квота уже израсходована."""

    bundle = make_runner(
        pipeline=RecordingPipeline(
            result=made_state(tmp_path, status="needs_review", pages=2)
        )
    )

    await bundle.runner.run(make_request())

    assert runner_module.PARTIAL_REDO not in bundle.bot.sent_texts
    assert runner_module.STORY_READY in bundle.bot.sent_texts


async def test_missing_page_still_refuses_quota_and_names_pages(
    tmp_path: Path,
) -> None:
    """Без одной картинки поведение прежнее: `FAILED`, номера и повтор."""

    state = made_state(
        tmp_path, status="needs_review", pages=3, illustrated=(1, 3)
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request(tale_id=13))

    summary = next(
        text
        for text in bundle.bot.sent_texts
        if text and text.startswith(runner_module.PARTIAL_TITLE)
    )
    assert "2" in summary
    assert runner_module.PARTIAL_REASON in summary
    update = bundle.repository.updates[-1]
    assert update["status"] is GenerationStatus.FAILED
    assert update["error"]
    assert runner_module.PARTIAL_REDO in bundle.bot.sent_texts


async def test_user_is_told_even_if_recording_fails(tmp_path: Path) -> None:
    """Сбой записи в БД не съедает сообщение пользователю.

    Запись идёт после отправки, поэтому сломанная БД не может оставить
    родителя в тишине после «Запускаю сказку!».
    """

    class BrokenRepository(FakeRepository):
        async def update_result(
            self, generation_id: int, **kwargs: Any
        ) -> None:
            raise RuntimeError("база недоступна")

    bundle = make_runner(
        pipeline=RecordingPipeline(
            result=made_state(
                tmp_path, status="needs_review", pages=2, illustrated=(1,)
            )
        ),
        repository=BrokenRepository(),
    )

    await bundle.runner.run(make_request())

    assert any(
        runner_module.PARTIAL_TITLE in (text or "")
        for text in bundle.bot.sent_texts
    )


async def test_vision_failure_message_survives_broken_database(
    tmp_path: Path,
) -> None:
    """То же для отказа по распознаванию игрушки."""

    from app.services.toy_vision import PermanentToyVisionError

    class BrokenRepository(FakeRepository):
        async def update_result(
            self, generation_id: int, **kwargs: Any
        ) -> None:
            raise RuntimeError("база недоступна")

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path)),
        vision=FakeVision(error=PermanentToyVisionError("не разглядел")),
        repository=BrokenRepository(),
    )

    await bundle.runner.run(make_request())

    assert bundle.bot.sent_texts == [runner_module.VISION_FAILED]


async def test_run_directory_contains_story_json(tmp_path: Path) -> None:
    """Каталог прогона получает `story.json` — как и в CLI (D19).

    Без него каталог бота остаётся набором картинок: ни сцен, ни вердиктов
    судьи, ни метрик этапов, а поддержка работает именно с каталогом.
    """

    settings = StorySettings(
        _env_file=None,  # type: ignore[call-arg]
        story_output_dir=str(tmp_path / "runs"),
    )
    bundle = make_runner(
        pipeline=RecordingPipeline(
            result=made_state(
                tmp_path, status="needs_review", pages=2, illustrated=(1,)
            )
        ),
        settings=settings,
    )

    await bundle.runner.run(make_request())

    run_dir = Path(bundle.repository.updates[-1]["output_dir"])
    payload = json.loads((run_dir / "story.json").read_text("utf-8"))
    assert payload["status"] == "needs_review"
    assert [page["page_num"] for page in payload["pages"]] == [1, 2]


async def test_story_json_failure_does_not_break_the_run(
    tmp_path: Path,
) -> None:
    """Незаписываемый `story.json` — не повод ронять выдачу результата."""

    settings = StorySettings(
        _env_file=None,  # type: ignore[call-arg]
        story_output_dir=str(tmp_path / "runs" / "page.png"),
    )
    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path, pages=2)),
        settings=settings,
    )

    await bundle.runner.run(make_request())

    assert bundle.repository.updates[-1]["status"] is GenerationStatus.SUCCESS
    assert runner_module.STORY_READY in bundle.bot.sent_texts


async def test_partial_result_offers_redo_with_restart_keyboard(
    tmp_path: Path,
) -> None:
    """Предложение повтора приходит с кнопкой «Начать заново»."""

    state = made_state(
        tmp_path, status="needs_review", pages=2, illustrated=(1,)
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    summary_items = [
        item
        for item in bundle.bot.sent
        if item.text and item.text.startswith(runner_module.PARTIAL_TITLE)
    ]
    redo_items = [
        item for item in bundle.bot.sent if item.text == runner_module.PARTIAL_REDO
    ]
    assert len(summary_items) == 1
    assert len(redo_items) == 1
    for item in (*summary_items, *redo_items):
        assert keyboard_labels(item.attachments) == [["Начать заново"]]


async def test_partial_result_is_not_counted_as_completed(
    tmp_path: Path,
) -> None:
    """Частичный прогон не засчитывается как завершённая сказка."""

    state = made_state(tmp_path, status="needs_review", pages=2, illustrated=(1,))
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request(tale_id=4))

    update = bundle.repository.updates[-1]
    assert update["generation_id"] == 4
    assert update["status"] is not GenerationStatus.SUCCESS
    assert update["error"]


async def test_partial_result_offers_redo_only_once(tmp_path: Path) -> None:
    """Повтор предлагается один раз, второй прогон молча уходит в выдачу."""

    state = made_state(tmp_path, status="needs_review", pages=2, illustrated=(1,))
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request(tale_id=1))
    assert bundle.bot.sent_texts.count(runner_module.PARTIAL_REDO) == 1

    bundle.bot.sent.clear()
    await bundle.runner.run(make_request(tale_id=2))
    assert bundle.bot.sent_texts.count(runner_module.PARTIAL_REDO) == 0
    assert any(
        runner_module.PARTIAL_TITLE in (text or "")
        for text in bundle.bot.sent_texts
    )


# --- Ограниченный набор иллюстрируемых страниц -----------------------------


async def test_limited_illustrations_are_not_treated_as_partial(
    tmp_path: Path,
) -> None:
    """Книжка «3 картинки из 6» доставлена целиком, а не частично.

    Регрессия на исходный дефект, который это изменение устраняет: пропуски
    считались по `1..num_pages`, поэтому страницы вне набора выглядели
    сбоем. Прогон уходил в ветку частичного результата, квота не
    расходовалась, а предложение «начать заново» квота разрешала — и
    бесплатная генерация становилась бесконечной.
    """

    state = made_state(
        tmp_path, pages=6, illustrated=(1, 4, 6), illustrated_set=(1, 4, 6)
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request(tale_id=21))

    texts = [text for text in bundle.bot.sent_texts if text]
    assert not any(text.startswith(runner_module.PARTIAL_TITLE) for text in texts)
    # Квота списана: весь набор иллюстрируемых страниц отдан.
    update = bundle.repository.updates[-1]
    assert update["status"] is GenerationStatus.SUCCESS
    # Повтор не предлагается — переделывать нечего.
    assert runner_module.PARTIAL_REDO not in bundle.bot.sent_texts


async def test_pages_out_of_set_come_as_plain_text(tmp_path: Path) -> None:
    """Страница вне набора уходит текстом с номером и без извинения.

    Извинение «картинку отрисовать не вышло» здесь было бы прямой ложью:
    картинку к этой странице не заказывали, её отсутствие заложено замыслом.
    """

    state = made_state(
        tmp_path, pages=4, illustrated=(1, 3), illustrated_set=(1, 3)
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    assert [item.text for item in page_messages(bundle.bot)] == [
        "Страница 1\nТекст страницы 1.",
        "Страница 2\nТекст страницы 2.",
        "Страница 3\nТекст страницы 3.",
        "Страница 4\nТекст страницы 4.",
    ]
    assert not any(
        runner_module.PAGE_IMAGE_SKIPPED in (item.text or "")
        for item in page_messages(bundle.bot)
    )


async def test_page_in_set_without_picture_still_explains_failure(
    tmp_path: Path,
) -> None:
    """Страница из набора без картинки сохраняет `PAGE_IMAGE_SKIPPED`.

    Это настоящий сбой: картинку заказывали, а получить не удалось. Родитель
    должен видеть причину дыры, иначе он не поймёт, что произошло.
    """

    state = made_state(
        tmp_path,
        status="needs_review",
        pages=4,
        illustrated=(1, 3),
        illustrated_set=(1, 2, 3),
    )
    state.metrics.append(
        StageMetrics(
            stage="illustrate",
            duration_ms=10.0,
            error="illustration: page 2 failed after retries: HTTP 429",
        )
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    texts = {item.text for item in page_messages(bundle.bot)}
    assert (
        f"Страница 2\nТекст страницы 2.\n\n"
        f"{runner_module.PAGE_IMAGE_SKIPPED}" in texts
    )
    # Страница вне набора извинения не получает.
    assert "Страница 4\nТекст страницы 4." in texts


async def test_illustration_note_names_run_pages_and_sends_once(
    tmp_path: Path,
) -> None:
    """Объяснение названо один раз и содержит номера фактического прогона."""

    state = made_state(
        tmp_path, pages=6, illustrated=(1, 4, 6), illustrated_set=(1, 4, 6)
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    notes = [
        text
        for text in bundle.bot.sent_texts
        if text and "Картинки нарисованы к страницам" in text
    ]
    assert len(notes) == 1
    assert "1, 4, 6" in notes[0]
    # Объяснение приходит до постраничной выдачи, а не после неё.
    texts = [text for text in bundle.bot.sent_texts if text]
    assert texts.index(notes[0]) < texts.index("Страница 1\nТекст страницы 1.")
    # Страницы без картинки объяснение не сопровождает.
    for item in page_messages(bundle.bot):
        assert "Картинки нарисованы к страницам" not in (item.text or "")


async def test_illustration_note_never_hardcodes_page_numbers(
    tmp_path: Path,
) -> None:
    """Номера в объяснении берутся из прогона, а не зашиты в текст.

    Зашитая тройка при `STORY_DEFAULT_PAGES = 8` стала бы прямой ложью:
    набор был бы (1, 6, 8), а пользователь прочёл бы «1, 4, 6».
    """

    state = made_state(
        tmp_path, pages=8, illustrated=(1, 6, 8), illustrated_set=(1, 6, 8)
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    note = next(
        text
        for text in bundle.bot.sent_texts
        if text and "Картинки нарисованы к страницам" in text
    )
    assert "1, 6, 8" in note
    assert "1, 4, 6" not in note


async def test_limited_illustrations_get_their_own_review_result(
    tmp_path: Path,
) -> None:
    """Итог с замечаниями не обещает иллюстраций у каждой страницы.

    `needs_review` ставится и при исчерпании бюджета ревизий, и тогда
    книжка доставлена целиком. При ограниченном наборе старый текст «все
    страницы с иллюстрациями» был бы ложью: родитель не получил картинок к
    страницам 2, 3, 5.
    """

    state = made_state(
        tmp_path,
        status="needs_review",
        pages=6,
        illustrated=(1, 4, 6),
        illustrated_set=(1, 4, 6),
    )
    state.metrics.append(
        StageMetrics(
            stage="quality",
            duration_ms=10.0,
            error="judge: passed=false after 2 revisions",
        )
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request(tale_id=22))

    summary = next(
        text
        for text in bundle.bot.sent_texts
        if text and text.startswith(runner_module.REVIEW_TITLE)
    )
    assert "бесплатного варианта" in summary
    assert "все с иллюстрациями" not in summary
    # Причина в чате — про текст, а не про отказ сервиса картинок.
    assert "отрисовать" not in summary
    update = bundle.repository.updates[-1]
    assert update["status"] is GenerationStatus.SUCCESS
    assert "judge: passed=false" in str(update["error"])


async def test_full_set_keeps_plain_review_result(tmp_path: Path) -> None:
    """При полном наборе итог прежний: все страницы с иллюстрациями."""

    state = made_state(
        tmp_path,
        status="needs_review",
        pages=2,
        illustrated=(1, 2),
        illustrated_set=(1, 2),
    )
    state.metrics.append(
        StageMetrics(
            stage="quality",
            duration_ms=10.0,
            error="judge: passed=false after 2 revisions",
        )
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    summary = next(
        text
        for text in bundle.bot.sent_texts
        if text and text.startswith(runner_module.REVIEW_TITLE)
    )
    assert "все с иллюстрациями" in summary
    assert "бесплатного варианта" not in summary


async def test_full_set_sends_no_illustration_note(tmp_path: Path) -> None:
    """При наборе, равном всем страницам, выдача выглядит как раньше."""

    state = made_state(
        tmp_path,
        pages=6,
        illustrated=(1, 2, 3, 4, 5, 6),
        illustrated_set=(1, 2, 3, 4, 5, 6),
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request())

    assert not any(
        "Картинки нарисованы к страницам" in (text or "")
        for text in bundle.bot.sent_texts
    )
    # Каждая страница пришла текстом вместе со своей картинкой.
    pages = page_messages(bundle.bot)
    assert len(pages) == 6
    assert all(item.attachments for item in pages)


async def test_real_failure_on_illustrated_page_keeps_partial_branch(
    tmp_path: Path,
) -> None:
    """Сбой на иллюстрируемой странице — это по-прежнему частичный результат.

    Регрессионный тест на семантику реального сбоя: ограничение не должно
    было размыть границу между «так задумано» и «не получилось». Прежнее
    поведение сохраняется целиком — `FAILED`, нерасходуемая квота, одно
    предложение переделать и итог про сбой, а не про бесплатный вариант.
    """

    state = made_state(
        tmp_path,
        status="needs_review",
        pages=6,
        illustrated=(1, 6),
        illustrated_set=(1, 4, 6),
    )
    state.metrics.append(
        StageMetrics(
            stage="illustrate",
            duration_ms=10.0,
            error=(
                "illustration: page 4 failed after retries: "
                "TransientImageGenerationError (HTTP 429)"
            ),
        )
    )
    bundle = make_runner(pipeline=RecordingPipeline(result=state))

    await bundle.runner.run(make_request(tale_id=23))

    texts = [text for text in bundle.bot.sent_texts if text]
    # Пропуск назван по набору: отсутствует именно страница 4, а не 2, 3, 5.
    summary = next(
        text for text in texts if text.startswith(runner_module.PARTIAL_TITLE)
    )
    assert "4" in summary
    assert runner_module.PARTIAL_REASON in summary
    # Итог описывает сбой, а не ограничение бесплатного варианта.
    assert not any(runner_module.REVIEW_TITLE in text for text in texts)
    assert not any("бесплатного варианта" in text for text in texts)
    # Квота не списана, предложение переделать — ровно одно.
    update = bundle.repository.updates[-1]
    assert update["status"] is GenerationStatus.FAILED
    assert update["error"]
    assert texts.count(runner_module.PARTIAL_REDO) == 1


# --- 5.8: остановка polling ---------------------------------------------------


async def test_shutdown_notifies_unfinished_runs(tmp_path: Path) -> None:
    """При остановке polling незавершённый прогон получает уведомление."""

    release = asyncio.Event()
    bundle = make_runner(
        pipeline=BlockingPipeline(
            result=made_state(tmp_path), release=release
        )
    )

    assert bundle.runner.start(make_request(chat_id=77)) is True
    await bundle.runner.shutdown()

    texts = [text for text in bundle.bot.sent_texts if text is not None]
    assert texts == [runner_module.RUN_INTERRUPTED]
    assert bundle.bot.sent_chat_ids == [77]
    assert bundle.runner.pending == 0


async def test_shutdown_without_runs_sends_nothing(tmp_path: Path) -> None:
    """Без незавершённых прогонов остановка никому ничего не пишет."""

    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path))
    )

    await bundle.runner.shutdown()

    assert bundle.bot.sent == []


async def test_vision_receives_photo_bytes(tmp_path: Path) -> None:
    """Распознавание получает те же байты фото, что и референс пайплайна."""

    vision = FakeVision()
    bundle = make_runner(
        pipeline=RecordingPipeline(result=made_state(tmp_path)),
        vision=vision,
    )
    photo = b"\x89PNG-photo-bytes"

    await bundle.runner.run(make_request(photo=photo))

    assert vision.photos == [photo]
