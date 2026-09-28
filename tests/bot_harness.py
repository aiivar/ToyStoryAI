"""Фейки MAX-событий и сервисов для тестов обработчиков."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from maxapi.types.attachments.image import Image

from app.bot.handlers import BotServices
from app.config import Settings
from app.db.repository import GenerationRepository
from app.services.photo_validation import PhotoValidator
from app.services.storage import PhotoStorage
from app.services.toy_vision import ToyDescription
from app.story_pipeline.models import StoryState
from app.story_pipeline.settings import StorySettings


class FakeMessage:
    """Мини-сообщение MAX, записывающее исходящие ответы."""

    def __init__(
        self,
        *,
        user_id: int = 1,
        chat_id: int = 1,
        attachments: list[Any] | None = None,
    ) -> None:
        self.sender = SimpleNamespace(user_id=user_id)
        self.recipient = SimpleNamespace(chat_id=chat_id, user_id=user_id)
        self.body = SimpleNamespace(attachments=attachments or [])
        self.sent: list[SimpleNamespace] = []

    async def answer(
        self,
        text: str | None = None,
        attachments: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.sent.append(
            SimpleNamespace(text=text, attachments=attachments)
        )

    @property
    def sent_texts(self) -> list[str | None]:
        return [item.text for item in self.sent]


class FakeMessageCreated:
    """Событие `message_created`."""

    def __init__(self, message: FakeMessage) -> None:
        self.message = message

    def get_ids(self) -> tuple[int | None, int | None]:
        return (
            self.message.recipient.chat_id,
            self.message.sender.user_id,
        )


class FakeBotStarted:
    """Событие `bot_started` (нажатие кнопки «Старт» в MAX)."""

    def __init__(self, *, user_id: int = 1, chat_id: int = 1) -> None:
        self.chat_id = chat_id
        self.user = SimpleNamespace(user_id=user_id)
        self.sent: list[SimpleNamespace] = []

    def get_ids(self) -> tuple[int, int]:
        return self.chat_id, self.user.user_id

    async def send(
        self,
        text: str | None = None,
        attachments: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.sent.append(SimpleNamespace(text=text, attachments=attachments))

    @property
    def sent_texts(self) -> list[str | None]:
        return [item.text for item in self.sent]


class FakeMessageCallback:
    """Событие `message_callback`."""

    def __init__(
        self,
        payload: str,
        *,
        user_id: int = 1,
        message: FakeMessage | None = None,
    ) -> None:
        self.callback = SimpleNamespace(
            payload=payload,
            user=SimpleNamespace(user_id=user_id),
        )
        self.message = message or FakeMessage(user_id=user_id)
        self.sent: list[SimpleNamespace] = []
        self.ack_count = 0
        self.answer_attachments: list[Any] = []

    async def ack(self, notification: str | None = None) -> None:
        raise AssertionError(
            "Пустой ack() запрещён: MAX требует message или notification."
        )

    async def answer(
        self,
        notification: str | None = None,
        new_text: str | None = None,
        attachments: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.ack_count += 1
        self.answer_attachments.append(attachments)

    async def send(
        self,
        text: str | None = None,
        attachments: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.sent.append(
            SimpleNamespace(text=text, attachments=attachments)
        )

    @property
    def sent_texts(self) -> list[str | None]:
        return [item.text for item in self.sent]


class FakeContext:
    """In-memory реализация FSM-контекста.

    `history` хранит снимки данных при каждом изменении: тестам нужно видеть,
    что лежало в состоянии в момент запуска генерации, — после сброса диалога
    это уже не проверить.
    """

    def __init__(
        self,
        state: Any | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.state = state
        self.data = dict(data or {})
        self.history: list[dict[str, Any]] = []

    def _snapshot(self) -> None:
        self.history.append(dict(self.data))

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def set_data(self, data: dict[str, Any]) -> None:
        self.data = dict(data)
        self._snapshot()

    async def update_data(self, **kwargs: Any) -> dict[str, Any]:
        self.data.update(kwargs)
        self._snapshot()
        return dict(self.data)

    async def set_state(self, state: Any | None = None) -> None:
        self.state = state
        self._snapshot()

    async def get_state(self) -> Any | None:
        return self.state

    async def clear(self) -> None:
        self.state = None
        self.data = {}
        self._snapshot()


class FakeDispatcher:
    """Диспетчер-заглушка, собирающий зарегистрированные обработчики."""

    def __init__(self) -> None:
        self.message_created_handlers: list[Any] = []
        self.message_callback_handlers: list[Any] = []
        self.bot_started_handlers: list[Any] = []

    def message_created(self, *args: Any, **kwargs: Any):
        def decorator(func: Any) -> Any:
            self.message_created_handlers.append(func)
            return func

        return decorator

    def bot_started(self, *args: Any, **kwargs: Any):
        def decorator(func: Any) -> Any:
            self.bot_started_handlers.append(func)
            return func

        return decorator

    def message_callback(self, *args: Any, **kwargs: Any):
        def decorator(func: Any) -> Any:
            self.message_callback_handlers.append(func)
            return func

        return decorator


class FakeSender:
    """Заглушка `ChatSender`: пишет исходящие сообщения по `chat_id`."""

    def __init__(self) -> None:
        self.sent: list[SimpleNamespace] = []

    async def send(
        self,
        chat_id: int,
        text: str | None = None,
        attachments: list[Any] | None = None,
    ) -> None:
        self.sent.append(
            SimpleNamespace(chat_id=chat_id, text=text, attachments=attachments)
        )

    @property
    def sent_texts(self) -> list[str | None]:
        return [item.text for item in self.sent]


class FakeVision:
    """Заглушка `ToyVisionService` с заранее заданным описанием игрушки."""

    def __init__(
        self,
        description: ToyDescription | None = None,
        error: Exception | None = None,
    ) -> None:
        self.description = description or ToyDescription(
            kind="плюшевый медвежонок",
            color="рыжий",
            material="мягкая ткань",
            features="кисточка на голове, красный шарф",
            short="рыжий плюшевый медвежонок в красном шарфе",
        )
        self.error = error
        self.photos: list[bytes] = []
        self.closed = 0

    async def describe(self, photo: bytes) -> ToyDescription:
        self.photos.append(photo)
        if self.error is not None:
            raise self.error
        return self.description

    async def aclose(self) -> None:
        self.closed += 1


@dataclass
class FakePipelineCall:
    """Аргументы, с которыми собрана заглушка пайплайна."""

    settings: Any
    reference_photo: bytes | None
    on_stage: Any


class FakePipeline:
    """Заглушка `StoryPipeline`: повторяет этапы и отдаёт заданное состояние."""

    def __init__(
        self,
        result: StoryState | None = None,
        error: Exception | None = None,
        stage_names: tuple[str, ...] = ("plan", "screenwrite", "illustrate"),
    ) -> None:
        self.result = result
        self.error = error
        self.stage_names = stage_names
        self.closed = 0
        self.stages: list[str] = []
        self.entered = asyncio.Event()
        # Аргументы сборки: тесты проверяют референс, настройки и колбэк.
        self.calls: list[FakePipelineCall] = []
        self._on_stage: Any = None

    def bind(self, call: FakePipelineCall) -> None:
        self.calls.append(call)
        self._on_stage = call.on_stage

    async def run(self, state: StoryState) -> StoryState:
        self.entered.set()
        if self._on_stage is not None:
            for name in self.stage_names:
                self.stages.append(name)
                await self._on_stage(name)
        if self.error is not None:
            raise self.error
        if self.result is None:  # pragma: no cover - опечатка в тесте
            raise AssertionError("Результат прогона не задан")
        return self.result

    async def aclose(self) -> None:
        self.closed += 1


def make_image_attachment(url: str) -> Image:
    """Создать вложение-изображение с заданным URL без валидации модели."""

    return Image.model_construct(
        type="image",
        payload=SimpleNamespace(url=url),
    )


def make_services(
    *,
    settings: Settings,
    repository: GenerationRepository,
    storage: PhotoStorage,
    sender: Any | None = None,
    vision: Any | None = None,
    pipeline: FakePipeline | None = None,
    story_settings: StorySettings | None = None,
    download_photo: Any | None = None,
) -> BotServices:
    """Собрать `BotServices` с фейковыми внешними сервисами.

    `pipeline` — заглушка пайплайна, состояние прогона которой задаёт тест.
    Заглучка по умолчанию результата не имеет и падает на `run`, поэтому тест,
    который доводит прогон до конца, обязан передать свою: пустой пайплайн
    годится только для проверок, где до `run` дело не доходит.
    """

    async def _download(url: str) -> bytes:  # pragma: no cover - подмена
        raise AssertionError("download_photo не должен вызываться в тесте")

    stub = pipeline if pipeline is not None else FakePipeline()

    def make_pipeline(
        *,
        settings: StorySettings,
        reference_photo: bytes | None,
        on_stage: Any,
    ) -> FakePipeline:
        stub.bind(
            FakePipelineCall(
                settings=settings,
                reference_photo=reference_photo,
                on_stage=on_stage,
            )
        )
        return stub

    return BotServices(
        settings=settings,
        validator=PhotoValidator(settings),
        storage=storage,
        repository=repository,
        download_photo=download_photo or _download,
        sender=sender or FakeSender(),
        vision=vision or FakeVision(),
        make_pipeline=make_pipeline,
        story_settings=story_settings or StorySettings(_env_file=None),  # type: ignore[call-arg]
    )


def keyboard_labels(attachments: list[Any] | None) -> list[list[str]]:
    """Извлечь подписи кнопок из вложения-клавиатуры."""

    if not attachments:
        return []
    markup = attachments[0]
    buttons = markup.payload.buttons
    return [[button.text for button in row] for row in buttons]


__all__ = [
    "FakeContext",
    "FakeDispatcher",
    "FakeMessage",
    "FakeMessageCallback",
    "FakeMessageCreated",
    "FakePipeline",
    "FakePipelineCall",
    "FakeSender",
    "FakeVision",
    "keyboard_labels",
    "make_image_attachment",
    "make_services",
]
