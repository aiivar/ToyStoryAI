"""Клавиатуры и наборы вариантов сценария сказки."""

from __future__ import annotations

from maxapi.types.attachments.buttons.callback_button import CallbackButton
from maxapi.utils.inline_keyboard import InlineKeyboardBuilder

# Наборы вариантов, заданные требованиями.
STYLE_OPTIONS: tuple[str, ...] = (
    "Волшебная",
    "Смешная",
    "Приключенческая",
)
ACTION_OPTIONS: tuple[str, ...] = (
    "Танец",
    "Полёт",
    "Путешествие",
)

# Payload'ы callback-кнопок.
UPLOAD_PHOTO_PAYLOAD = "upload_photo"
SUBSCRIBE_PAYLOAD = "subscribe"
RESTART_PAYLOAD = "restart"
STYLE_PREFIX = "style:"
ACTION_PREFIX = "action:"


def _option_payload(prefix: str, index: int) -> str:
    return f"{prefix}{index}"


def parse_style_payload(payload: str | None) -> str | None:
    """Вернуть стиль по payload'у кнопки или None."""

    return _parse_option(payload, STYLE_PREFIX, STYLE_OPTIONS)


def parse_action_payload(payload: str | None) -> str | None:
    """Вернуть действие по payload'у кнопки или None."""

    return _parse_option(payload, ACTION_PREFIX, ACTION_OPTIONS)


def _parse_option(
    payload: str | None,
    prefix: str,
    options: tuple[str, ...],
) -> str | None:
    if not payload or not payload.startswith(prefix):
        return None
    raw_index = payload[len(prefix) :]
    if not raw_index.isdigit():
        return None
    index = int(raw_index)
    if 0 <= index < len(options):
        return options[index]
    return None


def build_upload_photo_keyboard() -> InlineKeyboardBuilder:
    """Кнопка «Загрузить фото игрушки»."""

    builder = InlineKeyboardBuilder()
    builder.row(
        CallbackButton(
            text="Загрузить фото игрушки",
            payload=UPLOAD_PHOTO_PAYLOAD,
        )
    )
    return builder


def build_style_keyboard() -> InlineKeyboardBuilder:
    """Кнопки выбора стиля сказки."""

    builder = InlineKeyboardBuilder()
    for index, style in enumerate(STYLE_OPTIONS):
        builder.row(
            CallbackButton(
                text=style,
                payload=_option_payload(STYLE_PREFIX, index),
            )
        )
    return builder


def build_action_keyboard() -> InlineKeyboardBuilder:
    """Кнопки выбора действия игрушки."""

    builder = InlineKeyboardBuilder()
    for index, action in enumerate(ACTION_OPTIONS):
        builder.row(
            CallbackButton(
                text=action,
                payload=_option_payload(ACTION_PREFIX, index),
            )
        )
    return builder


def build_subscription_keyboard() -> InlineKeyboardBuilder:
    """Кнопка предложения подписки для исчерпанной квоты."""

    builder = InlineKeyboardBuilder()
    builder.row(
        CallbackButton(
            text="Оформить подписку",
            payload=SUBSCRIBE_PAYLOAD,
        )
    )
    return builder


def build_restart_keyboard() -> InlineKeyboardBuilder:
    """Кнопка «Начать заново» после ошибки."""

    builder = InlineKeyboardBuilder()
    builder.row(
        CallbackButton(
            text="Начать заново",
            payload=RESTART_PAYLOAD,
        )
    )
    return builder


__all__ = [
    "ACTION_OPTIONS",
    "ACTION_PREFIX",
    "RESTART_PAYLOAD",
    "STYLE_OPTIONS",
    "STYLE_PREFIX",
    "SUBSCRIBE_PAYLOAD",
    "UPLOAD_PHOTO_PAYLOAD",
    "build_action_keyboard",
    "build_restart_keyboard",
    "build_style_keyboard",
    "build_subscription_keyboard",
    "build_upload_photo_keyboard",
    "parse_action_payload",
    "parse_style_payload",
]
