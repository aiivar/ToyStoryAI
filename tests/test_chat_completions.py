"""Тесты общих правил Chat Completions (`app/utils/chat_completions.py`).

Правила вынесены, потому что их используют два клиента к одному вендору.
Особенно важна таблица статусов: 429 и 5xx формально тоже `>= 400`, поэтому
проверяется именно классификация, а не отдельные предикаты.
"""

from __future__ import annotations

import json

import pytest

from app.utils.chat_completions import (
    PERMANENT_STATUS_CODES,
    StatusCategory,
    classify_status,
    parse_json_object,
    strip_code_fence,
)


# --- Обёртка ответа ---------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param('{"a": 1}', '{"a": 1}', id="plain"),
        pytest.param('  {"a": 1}  ', '{"a": 1}', id="surrounded"),
        pytest.param('```json\n{"a": 1}\n```', '{"a": 1}', id="json-fence"),
        pytest.param('```\n{"a": 1}\n```', '{"a": 1}', id="bare-fence"),
        pytest.param('```JSON\n{"a": 1}\n```', '{"a": 1}', id="uppercase-tag"),
        pytest.param('```py\n{"a": 1}\n```', '{"a": 1}', id="foreign-lang"),
    ],
)
def test_strip_code_fence(content: str, expected: str) -> None:
    """Markdown-обёртка снимается, чистый JSON не меняется."""

    assert strip_code_fence(content) == expected


def test_parse_json_object_reads_fenced_payload() -> None:
    """Разбор ответа сочетает снятие обёртки и загрузку JSON."""

    assert parse_json_object('```json\n{"kind": "мишка"}\n```') == {
        "kind": "мишка"
    }
    assert parse_json_object('{"kind": "мишка"}') == {"kind": "мишка"}


def test_parse_json_object_raises_on_garbage() -> None:
    """Мусор на входе даёт `json.JSONDecodeError`, а не тихой пустой объект."""

    with pytest.raises(json.JSONDecodeError):
        parse_json_object("не JSON вовсе")


# --- Таблица статусов -------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(200, StatusCategory.OK, id="ok"),
        pytest.param(204, StatusCategory.OK, id="no-content"),
        pytest.param(400, StatusCategory.PERMANENT, id="bad-request"),
        pytest.param(401, StatusCategory.PERMANENT, id="unauthorized"),
        pytest.param(403, StatusCategory.PERMANENT, id="forbidden"),
        pytest.param(404, StatusCategory.PERMANENT, id="not-found"),
        pytest.param(422, StatusCategory.PERMANENT, id="unprocessable"),
        pytest.param(418, StatusCategory.PERMANENT, id="other-4xx"),
        pytest.param(429, StatusCategory.TRANSIENT, id="rate-limit"),
        pytest.param(500, StatusCategory.TRANSIENT, id="server-error"),
        pytest.param(503, StatusCategory.TRANSIENT, id="unavailable"),
    ],
)
def test_classify_status(status: int, expected: StatusCategory) -> None:
    """Статус однозначно раскладывается на постоянный, временный или успех."""

    assert classify_status(status) is expected


def test_transient_statuses_are_not_permanent() -> None:
    """429 и 5xx не должны попадать в постоянные: повтор там осмыслен."""

    assert 429 not in PERMANENT_STATUS_CODES
    for status in (429, 500, 502, 503):
        assert classify_status(status) is StatusCategory.TRANSIENT
