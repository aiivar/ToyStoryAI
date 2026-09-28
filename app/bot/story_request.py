"""Детерминированная сборка входа пайплайна: тема, описание, каталог прогона.

Тема сказки и описание игрушки-героя собираются здесь из распознанного
`ToyDescription`, выбранной морали и возраста. Сборка строковая и без
сети: модель не участвует в формулировке темы, поэтому один и тот же вход
всегда даёт один и тот же вход прогона, а распознавание игрушки остаётся
единственным местом, где текст приходит от LLM.

Каталог прогона — `output/{user_id}-{метка времени}/`: уникальный на
пользователя и момент запуска. Фоновые задачи делают параллельные прогоны
нормой, а общий каталог означал бы, что `page_1.png` одного ребёнка
перезапишет файл другого.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from app.services.toy_vision import ToyDescription
from app.story_pipeline.models import PageText
from app.story_pipeline.settings import StorySettings

# Метка времени в имени каталога: сортируется лексикографически и совпадает
# по смыслу с меткой CLI-прогона. Микросекунды добавлены, чтобы два прогона
# одного пользователя в одну секунду не получили один каталог.
_RUN_DIR_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S-%f"


def build_toy_description(description: ToyDescription) -> str:
    """Собрать строку описания игрушки из полей распознавания.

    Порядок и формулировки фиксированы: описание читают планировщик,
    экстрактор и судья, и расхождение между ними началось бы с разной
    сборки одного и того же объекта.
    """

    return (
        f"Вид: {description.kind}. "
        f"Цвет: {description.color}. "
        f"Материал: {description.material}. "
        f"Особенности: {description.features}."
    )


def build_theme(description: ToyDescription, moral: str, age: int) -> str:
    """Собрать тему сказки из описания игрушки, морали и возраста.

    Мораль подставляется дословно, без переформулирования: курируемый список
    уже написан так, чтобы быть моралью сказки, а не подсказкой для модели.
    """

    return (
        f"Сказка о любимой игрушке: {description.short}. "
        "Главный герой — эта игрушка, и её облик не меняется: "
        f"{build_toy_description(description)} "
        f"Мораль сказки: {moral}. "
        f"Читатель: ребёнок {age} лет, простые короткие предложения."
    )


def build_run_dir(
    user_id: int,
    settings: StorySettings,
    now: datetime | None = None,
) -> str:
    """Каталог результатов конкретного прогона бота.

    Корень берётся из `story_output_dir`, чтобы настройка каталога
    работала и для бота; дальше добавляются идентификатор пользователя и
    метка времени.
    """

    moment = now or datetime.now(timezone.utc)
    stamp = moment.strftime(_RUN_DIR_TIMESTAMP_FORMAT)
    return f"{settings.story_output_dir}/{user_id}-{stamp}"


def join_pages_text(pages: Sequence[PageText]) -> str:
    """Склеить тексты страниц в один абзац в порядке номеров страниц.

    Такой текст пишется в запись о генерации: по нему видно, что именно
    получилось, без чтения картинок из каталога прогона.
    """

    ordered = sorted(pages, key=lambda page: page.page_num)
    return " ".join(page.text.strip() for page in ordered if page.text.strip())


__all__ = [
    "build_run_dir",
    "build_theme",
    "build_toy_description",
    "join_pages_text",
]
