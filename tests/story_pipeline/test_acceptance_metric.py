"""Приёмочный помощник метрики качества (задача 11.3).

Считает долю прогонов, где судья прошёл проверку с первой попытки без единой
ревизии, — целевой показатель не ниже 70% (критерий приёмки №3). Логика
вычисления — чистые функции, они проверяются герметично. Серия живых
прогонов вынесена отдельным тестом под маркером `integration`: по умолчанию
она деселектится (`pytest.ini`, `-m "not integration"`) и требует реального
`DEEPSEEK_API_KEY`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from app.story_pipeline import cli
from app.story_pipeline.models import JudgeVerdict, StoryState
from app.story_pipeline.settings import StorySettings, get_story_settings

# Разумный дефолт серии: одной-двух сказок мало для оценки доли, а десяток
# даёт устойчивую оценку при умеренном расходе текстовых вызовов DeepSeek.
DEFAULT_RUNS = 10

# Целевой показатель критерия приёмки №3: не ниже 70% без ревизии.
PASS_RATE_THRESHOLD = 0.70

# Параметры живого прогона: реалистичная детская сказка среднего возраста.
LIVE_THEME = "Приключение плюшевого мишки в лесу"
LIVE_AGE = 5
LIVE_MORAL = "Смелость помогает найти друзей"


def passed_first_try(state: StoryState) -> bool:
    """Прошёл ли прогон судью с первой попытки и без единой ревизии.

    Засчитывается только вердикт самого первого судейства при нулевом
    `revision_count`: если петля контроля качества хоть раз позвала
    ревизора (в том числе из-за остатка неизвестных имён страниц), прогон
    уже не «прошёл без ревизии», даже если итоговый вердикт положительный.
    """

    if state.revision_count != 0 or not state.judge_history:
        return False
    return state.judge_history[0].passed


def first_try_pass_rate(
    states: Sequence[StoryState],
    *,
    failures: int = 0,
) -> float:
    """Доля прогонов, прошедших судью с первой попытки (значение 0..1).

    `failures` — число прогонов, упавших до вынесения вердикта (ненулевой
    код возврата CLI). Они входят в знаменатель как не прошедшие: иначе
    сломанная серия, где половина прогонов падает, показывала бы 100% и
    ложно проходила приёмку. Ошибка вызова — только полностью пустая серия
    (`states` пусто и `failures == 0`): доля от нуля прогонов не определена.
    """

    total = len(states) + failures
    if total == 0:
        raise ValueError("Нельзя вычислить долю прохождения по пустой серии.")
    passed = sum(1 for state in states if passed_first_try(state))
    return passed / total


def _state(
    *,
    revision_count: int = 0,
    verdicts: Sequence[bool] = (),
) -> StoryState:
    """Собрать минимальное состояние с нужной историей вердиктов."""

    return StoryState(
        theme="Тема",
        age=5,
        moral="Мораль",
        num_pages=2,
        revision_count=revision_count,
        judge_history=[
            JudgeVerdict(passed=passed, issues=[], problem_pages=[])
            for passed in verdicts
        ],
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


def test_first_try_pass_true_for_passed_without_revisions() -> None:
    """Судья прошёл сразу, ревизий не было — засчитываем."""

    assert passed_first_try(_state(verdicts=[True])) is True


def test_first_try_pass_false_after_revision() -> None:
    """Судья прошёл, но уже после ревизии — первая попытка не засчитана."""

    state = _state(revision_count=1, verdicts=[False, True])

    assert passed_first_try(state) is False


def test_first_try_pass_false_when_judge_failed_first() -> None:
    """Первый вердикт отрицательный — прогон не прошёл без ревизии."""

    assert passed_first_try(_state(verdicts=[False])) is False


def test_first_try_pass_false_without_verdicts() -> None:
    """Прогон без вердиктов судьи не считается прошедшим."""

    assert passed_first_try(_state()) is False


def test_first_try_pass_rate_counts_passed_runs() -> None:
    """Доля считается по всем состояниям серии."""

    states = [
        _state(verdicts=[True]),
        _state(revision_count=1, verdicts=[False, True]),
        _state(verdicts=[True]),
        _state(verdicts=[True]),
    ]

    assert first_try_pass_rate(states) == 0.75


def test_first_try_pass_rate_reaches_threshold_at_seventy_percent() -> None:
    """Ровно 70% (7 из 10) удовлетворяет целевому показателю."""

    states = [_state(verdicts=[True]) for _ in range(7)]
    states += [
        _state(revision_count=1, verdicts=[False, True]) for _ in range(3)
    ]

    assert first_try_pass_rate(states) == pytest.approx(PASS_RATE_THRESHOLD)
    assert first_try_pass_rate(states) >= PASS_RATE_THRESHOLD


def test_first_try_pass_rate_counts_crashes_as_non_passing() -> None:
    """Сбойные прогоны входят в знаменатель и снижают долю, а не исчезают."""

    states = [_state(verdicts=[True]), _state(verdicts=[True])]

    assert first_try_pass_rate(states, failures=2) == 0.5


def test_broken_series_cannot_report_full_pass_rate() -> None:
    """Серия, где половина прогонов упала, не может показать 100%."""

    states = [_state(verdicts=[True]) for _ in range(5)]

    rate = first_try_pass_rate(states, failures=5)

    assert rate == 0.5
    assert rate < PASS_RATE_THRESHOLD


def test_all_crashed_series_has_zero_rate() -> None:
    """Если упали все прогоны, доля нулевая, а не неопределённая."""

    assert first_try_pass_rate([], failures=5) == 0.0


def test_first_try_pass_rate_rejects_empty_series() -> None:
    """Пустая серия не имеет доли — это ошибка вызова, не ноль."""

    with pytest.raises(ValueError):
        first_try_pass_rate([])


def _live_run_state(
    output_root: Path,
    settings: StorySettings,
    *,
    index: int,
) -> tuple[int, StoryState | None]:
    """Один живой прогон CLI: код возврата и прочитанное состояние."""

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run_settings = settings.model_copy(
        update={
            "story_output_dir": str(output_root),
            # Метрике нужны только вердикты судьи: изображения мокаются,
            # чтобы серия не тратила деньги на генерацию картинок.
            "gemini_api_key": "",
        }
    )
    stamp = (base + timedelta(seconds=index)).strftime("%Y%m%d-%H%M%S")

    code = cli.run_cli(
        [
            "--theme",
            LIVE_THEME,
            "--age",
            str(LIVE_AGE),
            "--moral",
            LIVE_MORAL,
        ],
        settings=run_settings,
        now=lambda: base + timedelta(seconds=index),
    )
    if code != 0:
        return code, None
    payload = json.loads(
        (output_root / stamp / "story.json").read_text("utf-8")
    )
    return code, StoryState.model_validate(payload)


def measure_live_first_try_pass_rate(
    runs: int,
) -> tuple[list[StoryState], int]:
    """Прогнать CLI `runs` раз и вернуть состояния и число сбоев.

    Сбоем считается ненулевой код возврата: такое состояние в список не
    попадает (судья его не выносил), но число сбоев возвращается отдельно,
    чтобы вызывающий включил их в знаменатель метрики — упавший прогон
    понижает долю, а не исчезает из неё.
    """

    output_root = Path(".tmp/tests") / f"story-acceptance-{uuid4().hex}"
    settings = get_story_settings()
    states: list[StoryState] = []
    failures = 0
    for index in range(runs):
        code, state = _live_run_state(output_root, settings, index=index)
        if state is None:
            failures += 1
            continue
        states.append(state)
    return states, failures


@pytest.mark.integration
def test_live_first_try_pass_rate_meets_threshold() -> None:
    """Серия живых прогонов: судья без ревизии не реже, чем в 70% случаев."""

    settings = get_story_settings()
    if not settings.deepseek_api_key.strip():
        pytest.skip(
            "DEEPSEEK_API_KEY не задан: живая серия прогонов невозможна."
        )

    states, failures = measure_live_first_try_pass_rate(DEFAULT_RUNS)
    passed = sum(1 for state in states if passed_first_try(state))
    # Сбои идут в знаменатель: сломанная серия понижает долю и не может
    # показать 100% при половине упавших прогонов.
    rate = first_try_pass_rate(states, failures=failures)
    print(
        f"\nЖивая серия: сбоев {failures}, прогонов {DEFAULT_RUNS}, "
        f"успешных {len(states)}, без ревизии {passed}, доля {rate:.0%}"
    )

    assert rate >= PASS_RATE_THRESHOLD, (
        f"Доля прохождения без ревизии {rate:.0%} ниже "
        f"целевых {PASS_RATE_THRESHOLD:.0%} "
        f"(сбоев {failures} из {DEFAULT_RUNS})."
    )
