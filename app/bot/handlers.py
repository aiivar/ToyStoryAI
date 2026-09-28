"""Обработчики событий MAX: диалог выбора параметров и запуск генерации.

Обработчики — тонкие адаптеры: разбирают событие, переключают состояния FSM
и передают запрос в `StoryRunner`. Ни HTTP, ни бизнес-правил здесь нет:
всё, что связано с генерацией, выдачей страниц и записью в БД, живёт в
`app/bot/runner.py`, внешний I/O — в `app/services/*`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from maxapi import F
from maxapi.context.base import BaseContext
from maxapi.filters.command import CommandStart
from maxapi.types.updates.message_callback import MessageCallback
from maxapi.types.updates.message_created import MessageCreated

from app.bot.keyboards import (
    AGE_PREFIX,
    MORAL_PREFIX,
    RESTART_PAYLOAD,
    SUBSCRIBE_PAYLOAD,
    UPLOAD_PHOTO_PAYLOAD,
    build_age_keyboard,
    build_moral_keyboard,
    build_restart_keyboard,
    build_subscription_keyboard,
    build_upload_photo_keyboard,
    parse_age_payload,
    parse_moral_payload,
)
from app.bot.messaging import ChatSender
from app.bot.runner import PipelineFactory, RunRequest, StoryRunner
from app.bot.states import StoryFlow
from app.config import Settings
from app.db.repository import GenerationRepository
from app.services.photo_validation import PhotoValidationError, PhotoValidator
from app.services.storage import PhotoStorage, PhotoStorageError
from app.services.toy_vision import ToyVisionService
from app.story_pipeline.settings import StorySettings
from app.utils.logger import get_logger

logger = get_logger()

# --- Тексты сообщений (пользовательские строки на русском) --------------------

GREETING = (
    "Привет! Я превращаю фотографию любимой игрушки в персональную "
    "сказку с картинками — по одной странице в сообщении.\n\n"
    "Как это работает:\n"
    "1. Пришлите фото игрушки (JPG, PNG или WebP, до 10 МБ).\n"
    "2. Выберите возраст ребёнка и мораль сказки.\n"
    "3. Получите готовую сказку прямо в чате.\n\n"
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
PHOTO_UNREADABLE = (
    "Не получилось прочитать сохранённое фото. Пришлите его ещё раз."
)
AGE_PROMPT = "Сколько лет ребёнку?"
MORAL_PROMPT = "О чём мораль сказки?"
AGE_HINT = "Выберите возраст кнопкой ниже — так сказка получится подходящей."
MORAL_HINT = "Выберите мораль кнопкой ниже — я подставлю её в сказку."
UNKNOWN_OPTION = (
    "Такого варианта нет. Выберите один из предложенных на кнопках."
)
GENERATION_STARTED = (
    "Запускаю сказку! Сообщу, как пойдёт дело, а потом пришлю страницы "
    "с картинками — по одной в сообщении."
)
ALREADY_RUNNING = (
    "Сказка уже готовится, второй прогон не запускаю. Страницы придут "
    "сюда по мере готовности."
)
RESTART_HINT = "Давайте начнём заново. Пришлите фото игрушки."
SUBSCRIPTION_STUB = (
    "Покупка подписки пока недоступна — мы работаем над этим. "
    "Спасибо, что вы с нами!"
)

SendFunc = Callable[..., Awaitable[Any]]
DownloadFunc = Callable[[str], Awaitable[bytes]]


@dataclass
class BotServices:
    """Контейнер сервисов, внедряемых в обработчики.

    Отправитель привязан к `Bot`, а не к событию: событие MAX живёт один
    апдейт, а сообщения фоновой генерации уходят спустя минуты после него.
    `ToyVisionService` и фабрика пайплайна — внешние провайдеры, которые
    `StoryRunner` собирает в прогон; здесь они нужны, чтобы подменять их
    фейками в тестах.
    """

    settings: Settings
    validator: PhotoValidator
    storage: PhotoStorage
    repository: GenerationRepository
    download_photo: DownloadFunc
    sender: ChatSender
    vision: ToyVisionService
    make_pipeline: PipelineFactory
    story_settings: StorySettings


@dataclass
class Handlers:
    """Ссылки на зарегистрированные обработчики и их сборку (нужны тестам)."""

    start: Callable[..., Awaitable[None]]
    bot_started: Callable[..., Awaitable[None]]
    photo: Callable[..., Awaitable[None]]
    other_message: Callable[..., Awaitable[None]]
    callback: Callable[..., Awaitable[None]]
    runs: StoryRunner


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


def _callback_user_id(event: MessageCallback) -> int:
    if event.callback is not None:
        return event.callback.user.user_id
    raise ValueError("Callback без пользователя.")  # pragma: no cover


def _callback_chat_id(event: MessageCallback, user_id: int) -> int:
    """Идентификатор чата для фоновой отправки.

    В личном диалоге он совпадает с `user_id`, но адресат берётся из
    получателя сообщения: так отправка переживает смену формата чата и не
    зависит от того, что событие ещё живо.
    """

    message = event.message
    recipient = getattr(message, "recipient", None)
    chat_id = getattr(recipient, "chat_id", None)
    return chat_id if isinstance(chat_id, int) else user_id


async def _send_greeting(send: SendFunc, context: BaseContext) -> None:
    """Сбросить диалог и отправить приветствие с кнопкой загрузки фото."""

    await context.clear()
    await context.set_state(StoryFlow.waiting_photo)
    await _reply(
        send,
        GREETING,
        [build_upload_photo_keyboard().as_markup()],
    )


# --- Регистрация обработчиков -------------------------------------------------


def register_handlers(dp: Any, services: BotServices) -> Handlers:
    """Зарегистрировать обработчики на диспетчере и вернуть их для тестов."""

    runs = StoryRunner(
        repository=services.repository,
        vision=services.vision,
        make_pipeline=services.make_pipeline,
        story_settings=services.story_settings,
        sender=services.sender,
    )

    async def on_start(event: MessageCreated, context: BaseContext) -> None:
        await _send_greeting(event.message.answer, context)

    async def on_bot_started(event: Any, context: BaseContext) -> None:
        # Кнопка «Старт» в MAX присылает bot_started, а не сообщение /start.
        await _send_greeting(event.send, context)

    async def on_photo(event: MessageCreated, context: BaseContext) -> None:
        user_id = _extract_user_id(event)
        if runs.is_running(user_id):
            # Идёт платный прогон: вторую сказку не начинаем, иначе один
            # нетерпеливый пользователь оплачивал бы прогоны пачками.
            logger.info("Фото во время прогона пользователя {}", user_id)
            await _reply(event.message.answer, ALREADY_RUNNING)
            return

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
        await context.set_state(StoryFlow.waiting_age)
        await _reply(
            event.message.answer,
            AGE_PROMPT,
            [build_age_keyboard().as_markup()],
        )

    async def on_other_message(
        event: MessageCreated,
        context: BaseContext,
    ) -> None:
        """Ответить на неожиданное сообщение подсказкой с нужной клавиатурой.

        Молчание здесь хуже подсказки: пользователь не понимает, что бот
        ждёт кнопку, и пишет ещё текст. Пока идёт платный прогон, на любое
        сообщение отвечаем одинаково: идёт генерация, второй раз не запускаем.
        """

        if runs.is_running(_extract_user_id(event)):
            await _reply(event.message.answer, ALREADY_RUNNING)
            return

        state = await context.get_state()
        if state == StoryFlow.waiting_age:
            await _reply(
                event.message.answer,
                AGE_HINT,
                [build_age_keyboard().as_markup()],
            )
            return
        if state == StoryFlow.waiting_moral:
            await _reply(
                event.message.answer,
                MORAL_HINT,
                [build_moral_keyboard().as_markup()],
            )
            return
        logger.info("Ожидалось фото, получено иное сообщение — прошу фото")
        await _reply(event.message.answer, NOT_A_PHOTO)

    async def on_age_callback(
        event: MessageCallback,
        context: BaseContext,
    ) -> None:
        """Принять возраст и спросить мораль."""

        age = parse_age_payload(_payload(event))
        if age is None:
            await _callback_send(
                event,
                UNKNOWN_OPTION,
                [build_age_keyboard().as_markup()],
            )
            return

        state = await context.get_state()
        if state != StoryFlow.waiting_age:
            await _callback_send(
                event,
                RESTART_HINT,
                [build_restart_keyboard().as_markup()],
            )
            return

        await context.update_data(age=age)
        await context.set_state(StoryFlow.waiting_moral)
        await _callback_send(
            event,
            MORAL_PROMPT,
            [build_moral_keyboard().as_markup()],
        )

    async def on_moral_callback(
        event: MessageCallback,
        context: BaseContext,
    ) -> None:
        """Принять мораль, запустить фоновую генерацию и сбросить диалог.

        Генерация уходит в фоновую задачу, а обработчик возвращается сразу:
        иначе dispatcher был бы занят минуту-полторы.
        """

        user_id = _callback_user_id(event)
        moral = parse_moral_payload(_payload(event))
        if moral is None:
            await _callback_send(
                event,
                UNKNOWN_OPTION,
                [build_moral_keyboard().as_markup()],
            )
            return

        if runs.is_running(user_id):
            logger.info("Повторный выбор морали во время прогона: {}", user_id)
            await _callback_send(event, ALREADY_RUNNING)
            return

        state = await context.get_state()
        data = await context.get_data()
        tale_id = data.get("tale_id")
        photo_path = data.get("photo_path")
        age = data.get("age")
        if (
            state != StoryFlow.waiting_moral
            or tale_id is None
            or not photo_path
            or not isinstance(age, int)
        ):
            await _callback_send(
                event,
                RESTART_HINT,
                [build_restart_keyboard().as_markup()],
            )
            return

        try:
            photo = await asyncio.to_thread(
                services.storage.read_bytes, photo_path
            )
        except PhotoStorageError as exc:
            logger.error("Фото прогона недоступно: {}", exc.reason)
            await _callback_send(event, PHOTO_UNREADABLE)
            return

        await context.update_data(moral=moral)

        request = RunRequest(
            user_id=user_id,
            chat_id=_callback_chat_id(event, user_id),
            tale_id=tale_id,
            photo_path=photo_path,
            photo=photo,
            age=age,
            moral=moral,
        )
        if not runs.start(request):
            await _callback_send(event, ALREADY_RUNNING)
            return

        await _callback_send(event, GENERATION_STARTED)
        # Диалог на этом закончен: параметры ушли в прогон, а состояние
        # снова ждёт фото. Отдельного состояния «генерируется» нет намеренно:
        # прогон живёт в фоне и дольше жизни апдейта, поэтому истина о том,
        # идёт ли он, берётся из реестра (`runs.is_running`), а не из FSM.
        # Ложь в состоянии оборачивается либо вечным «уже готовится», либо
        # возможностью запустить второй платный прогон.
        await context.clear()
        await context.set_state(StoryFlow.waiting_photo)

    async def on_callback(
        event: MessageCallback,
        context: BaseContext,
    ) -> None:
        await _answer_callback(event)
        payload = _payload(event)

        if payload == UPLOAD_PHOTO_PAYLOAD:
            await context.set_state(StoryFlow.waiting_photo)
            await _callback_send(event, PHOTO_PROMPT)
            return

        if payload is not None and payload.startswith(AGE_PREFIX):
            await on_age_callback(event, context)
            return

        if payload is not None and payload.startswith(MORAL_PREFIX):
            await on_moral_callback(event, context)
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

    dp.message_created(CommandStart())(on_start)
    dp.bot_started()(on_bot_started)
    dp.message_created(
        StoryFlow.waiting_photo, F.message.body.attachments
    )(on_photo)
    dp.message_created(
        StoryFlow.waiting_photo,
        StoryFlow.waiting_age,
        StoryFlow.waiting_moral,
    )(on_other_message)
    dp.message_callback()(on_callback)

    return Handlers(
        start=on_start,
        bot_started=on_bot_started,
        photo=on_photo,
        other_message=on_other_message,
        callback=on_callback,
        runs=runs,
    )


def _payload(event: MessageCallback) -> str | None:
    """Payload нажатой кнопки или None, если её не было."""

    return event.callback.payload if event.callback else None


__all__ = [
    "ALREADY_RUNNING",
    "AGE_HINT",
    "AGE_PROMPT",
    "BotServices",
    "GENERATION_STARTED",
    "GREETING",
    "Handlers",
    "MORAL_HINT",
    "MORAL_PROMPT",
    "NOT_A_PHOTO",
    "PHOTO_PROMPT",
    "RESTART_HINT",
    "SUBSCRIPTION_STUB",
    "UNKNOWN_OPTION",
    "register_handlers",
]
