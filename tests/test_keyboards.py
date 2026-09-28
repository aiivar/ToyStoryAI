"""Тесты состояний FSM и построителей клавиатур (задача 5.1)."""

from __future__ import annotations

from app.bot.keyboards import (
    AGE_OPTIONS,
    MORAL_OPTIONS,
    build_age_keyboard,
    build_moral_keyboard,
    build_restart_keyboard,
    build_subscription_keyboard,
    build_upload_photo_keyboard,
    parse_age_payload,
    parse_moral_payload,
)
from app.bot.states import StoryFlow
from tests.bot_harness import keyboard_labels

# Курируемый список моралей задан спецификацией и не подлежит правке
# разработчика: значения уходят в сказку дословно.
EXPECTED_MORALS = (
    "делиться с другими",
    "не сдаваться, а пробовать снова",
    "помогать тем, кому трудно",
    "вежливо просить и благодарить",
    "не бояться нового",
    "убирать за собой",
)


def test_age_options_are_three_to_six() -> None:
    """Клавиатура возраста предлагает ровно варианты 3, 4, 5 и 6."""

    assert AGE_OPTIONS == (3, 4, 5, 6)


def test_moral_options_match_specification() -> None:
    """Список моралей совпадает со списком из спецификации."""

    assert MORAL_OPTIONS == EXPECTED_MORALS


def test_age_keyboard_labels_and_payloads() -> None:
    """Клавиатура возраста содержит четыре кнопки с русскими подписями."""

    markup = build_age_keyboard().as_markup()
    buttons = markup.payload.buttons

    assert keyboard_labels([markup]) == [
        ["3 года"],
        ["4 года"],
        ["5 лет"],
        ["6 лет"],
    ]
    assert [button.payload for row in buttons for button in row] == [
        "age:3",
        "age:4",
        "age:5",
        "age:6",
    ]


def test_moral_keyboard_has_exact_morals_in_order() -> None:
    """Каждая кнопка морали подписана самой моралью, порядок задан списком."""

    markup = build_moral_keyboard().as_markup()
    buttons = markup.payload.buttons

    assert keyboard_labels([markup]) == [[moral] for moral in EXPECTED_MORALS]
    assert [button.payload for row in buttons for button in row] == [
        f"moral:{index}" for index in range(len(EXPECTED_MORALS))
    ]


def test_age_payload_round_trip() -> None:
    """Разбор payload кнопки возраста возвращает выбранный возраст."""

    for age in AGE_OPTIONS:
        assert parse_age_payload(f"age:{age}") == age


def test_moral_payload_round_trip() -> None:
    """Разбор payload кнопки морали возвращает мораль без изменений."""

    for index, moral in enumerate(MORAL_OPTIONS):
        assert parse_moral_payload(f"moral:{index}") == moral


def test_parse_payloads_reject_unknown_values() -> None:
    """Значение вне клавиатуры не разбирается — поддельный выбор отбрасывается."""

    assert parse_age_payload("age:9") is None
    assert parse_age_payload("age:-1") is None
    assert parse_age_payload("age:") is None
    assert parse_age_payload("moral:0") is None
    assert parse_age_payload(None) is None
    assert parse_moral_payload("moral:6") is None
    assert parse_moral_payload("moral:-1") is None
    assert parse_moral_payload("moral:x") is None
    assert parse_moral_payload("age:0") is None
    assert parse_moral_payload(None) is None


def test_service_keyboards_keep_single_buttons() -> None:
    """Служебные кнопки диалога не изменились."""

    assert keyboard_labels([build_upload_photo_keyboard().as_markup()]) == [
        ["Загрузить фото игрушки"]
    ]
    assert keyboard_labels([build_subscription_keyboard().as_markup()]) == [
        ["Оформить подписку"]
    ]
    assert keyboard_labels([build_restart_keyboard().as_markup()]) == [
        ["Начать заново"]
    ]


def test_story_flow_states_cover_new_dialog() -> None:
    """FSM знает состояния ожидания возраста и морали вместо стиля и действия."""

    # `StatesGroup` хранит состояния в атрибутах класса, а не в реестре
    # словаря: имена берутся из объявленных атрибутов, чтобы тест не
    # повторял внутреннее устройство `State`.
    from maxapi.context.state_machine import State

    names = {
        name
        for name, value in vars(StoryFlow).items()
        if isinstance(value, State)
    }

    assert {
        "waiting_photo",
        "waiting_age",
        "waiting_moral",
        "generating",
    } <= names
    assert "waiting_style" not in names
    assert "waiting_action" not in names
