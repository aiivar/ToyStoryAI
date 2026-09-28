"""Файловые артефакты прогона: полное состояние в `story.json`.

Артефакт один и тот же для обоих входов в пайплайн — CLI и бота, — поэтому и
писать его в одном месте: иначе каталог бота остался бы набором картинок без
сцен, вердиктов судьи и метрик этапов, а именно по ним разбор прогонов
и поддержка.

Файл не содержит бинарных данных (фото в прогон не попадают), поэтому
`StoryState` сериализуется в него целиком.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.story_pipeline.models import StoryState

# Имя файла полного состояния прогона.
STORY_FILENAME = "story.json"


def write_story_json(run_dir: str | Path, state: StoryState) -> None:
    """Записать полное состояние прогона в ``story.json``.

    Каталог создаётся здесь: в CLI к моменту записи его уже наполнил этап
    иллюстрирования, а в боте запись идёт после прогона, и каталог мог не
    пережить сбой на этапе рисования. `exist_ok` страхует оба случая.
    """

    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / STORY_FILENAME).write_text(
        json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = ["STORY_FILENAME", "write_story_json"]
