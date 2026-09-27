"""Фейки MAX-событий и сервисов для тестов обработчиков."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from maxapi.types.attachments.image import Image

from app.bot.handlers import BotServices
from app.config import Settings
from app.db.repository import GenerationRepository
from app.services.photo_validation import PhotoValidator
from app.services.storage import PhotoStorage
from app.services.video_provider import VideoJob, VideoResult


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
    """In-memory реализация FSM-контекста."""

    def __init__(
        self,
        state: Any | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.state = state
        self.data = dict(data or {})

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def set_data(self, data: dict[str, Any]) -> None:
        self.data = dict(data)

    async def update_data(self, **kwargs: Any) -> dict[str, Any]:
        self.data.update(kwargs)
        return dict(self.data)

    async def set_state(self, state: Any | None = None) -> None:
        self.state = state

    async def get_state(self) -> Any | None:
        return self.state

    async def clear(self) -> None:
        self.state = None
        self.data = {}


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


class FakeTextGenerator:
    """Заглушка генератора сказки."""

    def __init__(
        self,
        result: str = "Жила-была плюшевая игрушка...",
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def generate(
        self,
        *,
        style: str,
        action: str,
        toy_description: str | None = None,
    ) -> str:
        self.calls.append((style, action))
        if self.error is not None:
            raise self.error
        return self.result


class FakeVideoProvider:
    """Заглушка видеопровайдера."""

    def __init__(
        self,
        video_url: str = "https://cdn.example.com/video.mp4",
        create_error: Exception | None = None,
        wait_error: Exception | None = None,
    ) -> None:
        self.video_url = video_url
        self.create_error = create_error
        self.wait_error = wait_error
        self.created_with: list[tuple[str, str]] = []
        self.request_id = "req-fake"

    async def create_video(self, image_base64: str, action: str) -> VideoJob:
        if self.create_error is not None:
            raise self.create_error
        self.created_with.append((image_base64, action))
        return VideoJob(request_id=self.request_id)

    async def wait_for_video(self, request_id: str) -> VideoResult:
        if self.wait_error is not None:
            raise self.wait_error
        return VideoResult(video_url=self.video_url)


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
    text_generator: Any | None = None,
    video_provider: Any | None = None,
    download_photo: Any | None = None,
) -> BotServices:
    """Собрать `BotServices` с фейковыми внешними сервисами."""

    async def _download(url: str) -> bytes:  # pragma: no cover - подмена
        raise AssertionError("download_photo не должен вызываться в тесте")

    return BotServices(
        settings=settings,
        validator=PhotoValidator(settings),
        storage=storage,
        text_generator=text_generator or FakeTextGenerator(),
        video_provider=video_provider or FakeVideoProvider(),
        repository=repository,
        download_photo=download_photo or _download,
    )


def keyboard_labels(attachments: list[Any] | None) -> list[list[str]]:
    """Извлечь подписи кнопок из вложения-клавиатуры."""

    if not attachments:
        return []
    markup = attachments[0]
    buttons = markup.payload.buttons
    return [[button.text for button in row] for row in buttons]
