"""Обработчики событий MAX и оркестрация генерации.

Обработчики — тонкие адаптеры: они разбирают события, управляют переходами
FSM и вызывают сервисы из `BotServices`. Вся внешняя логика (DeepSeek, видео,
хранилище) живёт в `app/services/*`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from maxapi import F
from maxapi.context.base import BaseContext
from maxapi.filters.command import CommandStart
from maxapi.types.updates.message_callback import MessageCallback
from maxapi.types.updates.message_created import MessageCreated

from app.bot.keyboards import (
    RESTART_PAYLOAD,
    SUBSCRIBE_PAYLOAD,
    UPLOAD_PHOTO_PAYLOAD,
    build_action_keyboard,
    build_restart_keyboard,
    build_style_keyboard,
    build_subscription_keyboard,
    build_upload_photo_keyboard,
    parse_action_payload,
    parse_style_payload,
)
from app.bot.states import StoryFlow
from app.config import Settings
from app.db.models import GenerationStatus
from app.db.repository import GenerationRepository
from app.services.photo_validation import PhotoValidationError, PhotoValidator
from app.services.storage import PhotoStorage, PhotoStorageError
from app.services.text_generator import TextGenerationError, TextGenerator
from app.services.video_provider import VideoGenerationError, VideoProvider
from app.utils.logger import get_logger

logger = get_logger()

# --- Тексты сообщений (пользовательские строки на русском) --------------------

GREETING = (
    "Привет! Я превращаю фотографию любимой игрушки в персональную "
    "сказку и короткое видео.\n\n"
    "Как это работает:\n"
    "1. Пришлите фото игрушки (JPG или PNG, до 10 МБ).\n"
    "2. Выберите стиль сказки и действие игрушки.\n"
    "3. Получите сказку и видео прямо в чате.\n\n"
    "Нажмите кнопку ниже, чтобы начать."
)
PHOTO_PROMPT = (
    "Пришлите фотографию игрушки: JPG, PNG или WebP, до 10 МБ, "
    "не меньше 300×300 пикселей."
)
NOT_A_PHOTO = (
    "Нужна именно фотография игрушки. Пришлите её файлом в формате "
    "JPG, PNG или WebP."
)
STYLE_PROMPT = "Отличное фото! Какой стиль сказки выбрать?"
ACTION_PROMPT = "А какое действие должна выполнить игрушка?"
TALE_PROGRESS = "Генерирую сказку..."
VIDEO_PROGRESS = "Создаю видео (может занять 1–3 минуты)..."
TALE_FAILED = (
    "Не получилось сгенерировать сказку. Попробуйте начать заново."
)
VIDEO_UNAVAILABLE = (
    "Не удалось подготовить фото для видео, поэтому шаг с видео сейчас "
    "недоступен. Пришлите фото ещё раз."
)
VIDEO_FAILED = (
    "Сказка готова и остаётся у вас, но видео создать не удалось. "
    "Попробуйте позже."
)
FREE_LIMIT_REACHED = (
    "Вы уже использовали бесплатную генерацию: сказку и видео. "
    "Чтобы создать новую сказку, оформите подписку."
)
SUBSCRIPTION_STUB = (
    "Покупка подписки пока недоступна — мы работаем над этим. "
    "Спасибо, что вы с нами!"
)
RESTART_HINT = "Давайте начнём заново."

SendFunc = Callable[..., Awaitable[Any]]
DownloadFunc = Callable[[str], Awaitable[bytes]]


@dataclass
class BotServices:
    """Контейнер сервисов, внедряемых в обработчики."""

    settings: Settings
    validator: PhotoValidator
    storage: PhotoStorage
    text_generator: TextGenerator
    video_provider: VideoProvider
    repository: GenerationRepository
    download_photo: DownloadFunc


@dataclass
class Handlers:
    """Ссылки на зарегистрированные обработчики (нужны тестам)."""

    start: Callable[..., Awaitable[None]]
    bot_started: Callable[..., Awaitable[None]]
    photo: Callable[..., Awaitable[None]]
    other_message: Callable[..., Awaitable[None]]
    callback: Callable[..., Awaitable[None]]


# --- Вспомогательные функции --------------------------------------------------


def _extract_user_id(event: MessageCreated) -> int:
    sender = event.message.sender
    if sender is not None and sender.user_id is not None:
        return sender.user_id
    _, user_id = event.get_ids()
    if user_id is None:  # pragma: no cover - защита от неполного события
        raise ValueError("Не удалось определить пользователя события.")
    return user_id


def _attachment_type_value(attachment: Any) -> str | None:
    """Вернуть строковый тип вложения независимо от enum/строки."""

    raw_type = getattr(attachment, "type", None)
    value = getattr(raw_type, "value", raw_type)
    return value if isinstance(value, str) else None


def _extract_photo_url(event: MessageCreated) -> str | None:
    """Найти URL первого фото/файла во вложении сообщения.

    Определяем по строковому типу вложения, а не по классу модели: так
    устойчивее к тому, как `maxapi` разберёт union вложений.
    """

    body = event.message.body
    attachments = body.attachments if body else None
    if not attachments:
        return None

    for attachment in attachments:
        if _attachment_type_value(attachment) not in {"image", "file"}:
            continue
        payload = getattr(attachment, "payload", None)
        url = getattr(payload, "url", None)
        if isinstance(url, str) and url:
            return url
    return None


async def _reply(
    send: SendFunc,
    text: str,
    attachments: list[Any] | None = None,
) -> None:
    await send(text=text, attachments=attachments)


async def _callback_send(
    event: MessageCallback,
    text: str,
    attachments: list[Any] | None = None,
) -> None:
    if event.message is None:
        return
    await event.send(text=text, attachments=attachments)


async def _answer_callback(event: MessageCallback) -> None:
    """Подтвердить нажатие кнопки.

    MAX требует в ответе на callback либо `message`, либо `notification`;
    пустой `ack()` даёт ошибку `proto.payload`. `answer()` собирает ответ
    с исходным сообщением и потому принимается API.
    """

    if event.message is None:
        return
    await event.answer()


async def _send_greeting(send: SendFunc, context: BaseContext) -> None:
    """Сбросить диалог и отправить приветствие с кнопкой загрузки фото."""

    await context.clear()
    await context.set_state(StoryFlow.waiting_photo)
    await _reply(
        send,
        GREETING,
        [build_upload_photo_keyboard().as_markup()],
    )


# --- Оркестрация генерации ----------------------------------------------------


async def run_generation(
    *,
    send: SendFunc,
    services: BotServices,
    user_id: int,
    style: str,
    action: str,
    tale_id: int,
    photo_path: str,
) -> None:
    """Полный цикл: квота → сказка → видео → выдача результатов."""

    repository = services.repository

    if await repository.has_completed_full_generation(user_id):
        await _reply(
            send,
            FREE_LIMIT_REACHED,
            [build_subscription_keyboard().as_markup()],
        )
        return

    await _reply(send, TALE_PROGRESS)
    try:
        tale_text = await services.text_generator.generate(
            style=style,
            action=action,
        )
    except TextGenerationError as exc:
        logger.error("Сбой генерации сказки: {}", exc)
        await repository.update_result(
            tale_id,
            status=GenerationStatus.FAILED,
            error=str(exc),
        )
        await _reply(
            send,
            TALE_FAILED,
            [build_restart_keyboard().as_markup()],
        )
        return

    await repository.update_result(
        tale_id,
        status=GenerationStatus.SUCCESS,
        tale_text=tale_text,
    )
    await _reply(send, tale_text)

    await _reply(send, VIDEO_PROGRESS)
    try:
        image_base64 = services.storage.encode_base64(photo_path)
    except PhotoStorageError as exc:
        logger.error("Не удалось подготовить фото для видео: {}", exc)
        await _reply(send, VIDEO_UNAVAILABLE)
        return

    try:
        job = await services.video_provider.create_video(image_base64, action)
    except VideoGenerationError as exc:
        logger.error("Не удалось создать запрос видео: {}", exc)
        await _reply(
            send,
            VIDEO_FAILED,
            [build_restart_keyboard().as_markup()],
        )
        return

    video_record = await repository.create_video(
        user_id,
        request_id=job.request_id,
        local_path=photo_path,
    )

    try:
        result = await services.video_provider.wait_for_video(job.request_id)
    except VideoGenerationError as exc:
        logger.error("Сбой генерации видео: {}", exc)
        await repository.update_result(
            video_record.id,
            status=GenerationStatus.FAILED,
            error=str(exc),
        )
        await _reply(
            send,
            VIDEO_FAILED,
            [build_restart_keyboard().as_markup()],
        )
        return

    await repository.update_result(
        video_record.id,
        status=GenerationStatus.SUCCESS,
        video_url=result.video_url,
    )
    await _reply(send, f"Ваше видео готово:\n{result.video_url}")


# --- Регистрация обработчиков -------------------------------------------------


def register_handlers(dp: Any, services: BotServices) -> Handlers:
    """Зарегистрировать обработчики на диспетчере и вернуть их для тестов."""

    async def on_start(event: MessageCreated, context: BaseContext) -> None:
        await _send_greeting(event.message.answer, context)

    async def on_bot_started(event: Any, context: BaseContext) -> None:
        # Кнопка «Старт» в MAX присылает bot_started, а не сообщение /start.
        await _send_greeting(event.send, context)

    async def on_photo(event: MessageCreated, context: BaseContext) -> None:
        user_id = _extract_user_id(event)
        attachments = event.message.body.attachments if event.message.body else None
        logger.info(
            "Приём фото: вложений={} ({})",
            len(attachments) if attachments else 0,
            [
                f"{type(item).__name__}:{_attachment_type_value(item)}"
                for item in (attachments or [])
            ],
        )

        url = _extract_photo_url(event)
        if url is None:
            logger.warning("У вложения нет подходящего URL для скачивания")
            await _reply(event.message.answer, NOT_A_PHOTO)
            return

        try:
            data = await services.download_photo(url)
        except Exception as exc:  # noqa: BLE001 - внешний загрузчик MAX
            logger.error("Не удалось скачать фото: {}", exc)
            await _reply(event.message.answer, NOT_A_PHOTO)
            return

        logger.info("Фото скачано: {} байт", len(data))

        try:
            info = services.validator.validate(data)
        except PhotoValidationError as exc:
            logger.info(
                "Фото отклонено ({}): {}",
                exc.reason,
                exc.user_message,
            )
            await _reply(
                event.message.answer,
                f"{exc.user_message} Пришлите другое фото.",
            )
            return

        logger.info(
            "Фото принято: {} {}×{} ({} байт)",
            info.format,
            info.width,
            info.height,
            info.size_bytes,
        )

        path = services.storage.save(
            data,
            extension=info.extension,
            user_id=user_id,
        )
        tale = await services.repository.create_tale(
            user_id,
            local_path=str(path),
        )

        await context.update_data(
            tale_id=tale.id,
            photo_path=str(path),
        )
        await context.set_state(StoryFlow.waiting_style)
        await _reply(
            event.message.answer,
            STYLE_PROMPT,
            [build_style_keyboard().as_markup()],
        )

    async def on_other_message(
        event: MessageCreated,
        context: BaseContext,
    ) -> None:
        logger.info("Ожидалось фото, получено иное сообщение — прошу фото")
        await _reply(event.message.answer, NOT_A_PHOTO)

    async def on_callback(
        event: MessageCallback,
        context: BaseContext,
    ) -> None:
        await _answer_callback(event)
        payload = event.callback.payload if event.callback else None

        if payload == UPLOAD_PHOTO_PAYLOAD:
            await context.set_state(StoryFlow.waiting_photo)
            await _callback_send(event, PHOTO_PROMPT)
            return

        style = parse_style_payload(payload)
        if style is not None:
            state = await context.get_state()
            if state != StoryFlow.waiting_style:
                await _callback_send(event, RESTART_HINT)
                return
            await context.update_data(style=style)
            await context.set_state(StoryFlow.waiting_action)
            await _callback_send(
                event,
                ACTION_PROMPT,
                [build_action_keyboard().as_markup()],
            )
            return

        action = parse_action_payload(payload)
        if action is not None:
            state = await context.get_state()
            data = await context.get_data()
            style_value = data.get("style")
            tale_id = data.get("tale_id")
            photo_path = data.get("photo_path")

            if (
                state != StoryFlow.waiting_action
                or not style_value
                or tale_id is None
                or not photo_path
            ):
                await _callback_send(
                    event,
                    RESTART_HINT,
                    [build_restart_keyboard().as_markup()],
                )
                return

            await context.set_state(StoryFlow.generating)
            await run_generation(
                send=lambda text, attachments=None: _callback_send(
                    event, text, attachments
                ),
                services=services,
                user_id=_callback_user_id(event),
                style=style_value,
                action=action,
                tale_id=tale_id,
                photo_path=photo_path,
            )
            await context.clear()
            return

        if payload == SUBSCRIBE_PAYLOAD:
            await _callback_send(event, SUBSCRIPTION_STUB)
            return

        if payload == RESTART_PAYLOAD:
            await context.clear()
            await context.set_state(StoryFlow.waiting_photo)
            await _callback_send(
                event,
                GREETING,
                [build_upload_photo_keyboard().as_markup()],
            )
            return

    dp.message_created(CommandStart())(on_start)
    dp.bot_started()(on_bot_started)
    dp.message_created(
        StoryFlow.waiting_photo, F.message.body.attachments
    )(on_photo)
    dp.message_created(StoryFlow.waiting_photo)(on_other_message)
    dp.message_callback()(on_callback)

    return Handlers(
        start=on_start,
        bot_started=on_bot_started,
        photo=on_photo,
        other_message=on_other_message,
        callback=on_callback,
    )


def _callback_user_id(event: MessageCallback) -> int:
    if event.callback is not None:
        return event.callback.user.user_id
    raise ValueError("Callback без пользователя.")  # pragma: no cover


__all__ = [
    "BotServices",
    "Handlers",
    "register_handlers",
    "run_generation",
]
