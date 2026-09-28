"""Общие правила разбора ответа OpenAI-совместимого Chat Completions.

Два клиента проекта ходят в один и тот же вендорский API
(`app/story_pipeline/providers/llm.py` — этапы пайплайна,
`app/services/toy_vision.py` — распознавание игрушки), и оба обязаны
одинаково понимать две вещи, которые вендор меняет по-разному от вызова к
вызову:

* **обёртку ответа** — модель может вернуть JSON внутри блока кода
  ```` ```json … ``` ````, хотя и просили чистый объект;
* **таблицу статусов** — какие HTTP-коды означают «повтор бессмыслен, чини
  конфигурацию», а какие «попробуй ещё раз».

Если эти правила живут в двух файлах, они рано или поздно разъезжаются: в
одном клиенте добавят 408 в постоянные, а в другом забудут, и пайплайн начнёт
молча повторять запросы с неверным ключом. Поэтому правила вынесены сюда, а
таксономию ошибок каждый клиент по-прежнему держит свою: она отражает границу
владельца, а не формат ответа вендора.

Модуль лежит в `app/utils/` по прецеденту `image_mime.py`: он нужен двум
независимым подсистемам, не знает ни об одной из них и зависимостей не имеет.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Final

# Возможная markdown-обёртка вокруг JSON: ```json … ``` или ``` … ```.
_CODE_FENCE_RE = re.compile(
    r"^```[a-zA-Z0-9_-]*\s*(?P<body>.*?)\s*```$", re.DOTALL
)

# HTTP-статусы, при которых повтор бессмыслен: аутентификация и валидация.
# 400 и 422 сюда же попадают у вендоров, отклоняющих запрос с изображением.
PERMANENT_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {400, 401, 403, 404, 422}
)


class StatusCategory(StrEnum):
    """Как клиенту реагировать на HTTP-статус ответа."""

    OK = "ok"
    PERMANENT = "permanent"
    TRANSIENT = "transient"


def classify_status(status_code: int) -> StatusCategory:
    """Разложить HTTP-статус на категорию реакции.

    Одна функция вместо двух предикатов намеренно: 429 и 5xx тоже `>= 400`,
    поэтому порядок проверок легко перепутать и получить «повторяем
    аутентификацию». Здесь таблица единственная, и порядок задан один раз.
    """

    if status_code in PERMANENT_STATUS_CODES:
        return StatusCategory.PERMANENT
    if status_code == 429 or status_code >= 500:
        return StatusCategory.TRANSIENT
    if status_code >= 400:
        return StatusCategory.PERMANENT
    return StatusCategory.OK


def strip_code_fence(content: str) -> str:
    """Снять markdown-обёртку блока кода и вернуть полезную нагрузку.

    Текст без обёртки возвращается без изменений (с обрезкой краёв): правило
    должно быть безопасно и когда модель всё же ответила чистым JSON.
    """

    match = _CODE_FENCE_RE.match(content.strip())
    if match is None:
        return content.strip()
    return match.group("body").strip()


def parse_json_object(content: str) -> object:
    """Разобрать текст ответа в JSON, сняв возможную markdown-обёртку.

    Бросает `json.JSONDecodeError`, если разобрать не удалось: вызывающий код
    превращает это в свою типизированную ошибку, потому что таксономия у него
    своя.
    """

    return json.loads(strip_code_fence(content))


__all__ = [
    "PERMANENT_STATUS_CODES",
    "StatusCategory",
    "classify_status",
    "parse_json_object",
    "strip_code_fence",
]
