"""Клавиатуры диалога: возраст, мораль и служебные кнопки.

Списки вариантов заданы требованиями, а не придуманы интерфейсом: возраст
3–6 лет совпадает с допустимым диапазоном пайплайна (`story_age_min`/
`story_age_max`), а мораль — курируемый список из шести формулировок,
безопасных для детей и подходящих на роль морали сказки. Свободного ввода
нет: значение приходит только с кнопки, поэтому произвольный текст на входе
LLM невозможен, а поддельный payload разбирается в `None` и отбрасывается
вызывающим кодом.
"""

from __future__ import annotations

from typing import cast

from maxapi.types.attachments.buttons.callback_button import CallbackButton
from maxapi.utils.inline_keyboard import InlineKeyboardBuilder

from app.story_pipeline.models import Voice

# Возраст читателя: значения совпадают с диапазоном пайплайна по умолчанию.
AGE_OPTIONS: tuple[int, ...] = (3, 4, 5, 6)

# Курируемый список моралей. Формулировки родительские, поэтому в сказку
# уходят дословно: переписывать их «под глагол» нельзя, список и есть
# продуктовое требование.
MORAL_OPTIONS: tuple[str, ...] = (
    "делиться с другими",
    "не сдаваться, а пробовать снова",
    "помогать тем, кому трудно",
    "вежливо просить и благодарить",
    "не бояться нового",
    "убирать за собой",
)

# Подписи кнопок возраста: падеж после числительного, а не «3 года/лет».
_AGE_LABELS: dict[int, str] = {
    3: "3 года",
    4: "4 года",
    5: "5 лет",
    6: "6 лет",
}

# Payload'ы callback-кнопок.
UPLOAD_PHOTO_PAYLOAD = "upload_photo"
SUBSCRIBE_PAYLOAD = "subscribe"
RESTART_PAYLOAD = "restart"
AGE_PREFIX = "age:"
MORAL_PREFIX = "moral:"
VOICE_PREFIX = "voice:"

# Голоса озвучки: payload кодирует продуктовый выбор, а не имя голоса
# вендора. Имя модели и голоса живут в настройках, поэтому смена вендора не
# меняет клавиатуру и историю диалога.
VOICE_OPTIONS: dict[str, str] = {
    "female": "Женский голос",
    "male": "Мужской голос",
}


def age_label(age: int) -> str:
    """Подпись кнопки возраста; неизвестный возраст показывается числом."""

    return _AGE_LABELS.get(age, str(age))


def _option_payload(prefix: str, index: int) -> str:
    return f"{prefix}{index}"


def parse_age_payload(payload: str | None) -> int | None:
    """Вернуть возраст по payload'у кнопки или None.

    `None` означает «значения в клавиатуре нет»: и неверный формат, и число
    вне списка 3–6. Проверка по списку, а не по диапазону, закрывает
    поддельные payload вида `age:9` и `age:-1`.
    """

    if not payload or not payload.startswith(AGE_PREFIX):
        return None
    raw_value = payload[len(AGE_PREFIX) :]
    if not raw_value.isdigit():
        return None
    value = int(raw_value)
    return value if value in AGE_OPTIONS else None


def parse_moral_payload(payload: str | None) -> str | None:
    """Вернуть мораль по payload'у кнопки или None."""

    return _parse_option(payload, MORAL_PREFIX, MORAL_OPTIONS)


def parse_voice_payload(payload: str | None) -> Voice | None:
    """Вернуть голос по payload'у кнопки или None.

    Проверка идёт по белому списку `VOICE_OPTIONS`: поддельный payload
    `voice:woman` не должен превратиться в женский голос по остаточному
    принципу, иначе выбор пользователя перестаёт быть выбором.
    """

    if not payload or not payload.startswith(VOICE_PREFIX):
        return None
    raw_value = payload[len(VOICE_PREFIX) :]
    if raw_value not in VOICE_OPTIONS:
        return None
    return cast(Voice, raw_value)


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


def build_age_keyboard() -> InlineKeyboardBuilder:
    """Кнопки выбора возраста ребёнка (3–6 лет)."""

    builder = InlineKeyboardBuilder()
    for age in AGE_OPTIONS:
        builder.row(
            CallbackButton(
                text=age_label(age),
                payload=f"{AGE_PREFIX}{age}",
            )
        )
    return builder


def build_moral_keyboard() -> InlineKeyboardBuilder:
    """Кнопки выбора морали из курируемого списка."""

    builder = InlineKeyboardBuilder()
    for index, moral in enumerate(MORAL_OPTIONS):
        builder.row(
            CallbackButton(
                text=moral,
                payload=_option_payload(MORAL_PREFIX, index),
            )
        )
    return builder


def build_voice_keyboard() -> InlineKeyboardBuilder:
    """Кнопки выбора голоса озвучки: женский и мужской."""

    builder = InlineKeyboardBuilder()
    for value, label in VOICE_OPTIONS.items():
        builder.row(
            CallbackButton(
                text=label,
                payload=f"{VOICE_PREFIX}{value}",
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
    """Кнопка «Начать заново» после ошибки или частичного результата."""

    builder = InlineKeyboardBuilder()
    builder.row(
        CallbackButton(
            text="Начать заново",
            payload=RESTART_PAYLOAD,
        )
    )
    return builder


__all__ = [
    "AGE_OPTIONS",
    "AGE_PREFIX",
    "MORAL_OPTIONS",
    "MORAL_PREFIX",
    "RESTART_PAYLOAD",
    "SUBSCRIBE_PAYLOAD",
    "UPLOAD_PHOTO_PAYLOAD",
    "VOICE_OPTIONS",
    "VOICE_PREFIX",
    "age_label",
    "build_age_keyboard",
    "build_moral_keyboard",
    "build_restart_keyboard",
    "build_subscription_keyboard",
    "build_upload_photo_keyboard",
    "build_voice_keyboard",
    "parse_age_payload",
    "parse_moral_payload",
    "parse_voice_payload",
]
