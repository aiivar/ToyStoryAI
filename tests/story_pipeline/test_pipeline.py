"""Сквозные тесты оркестратора пайплайна (`app/story_pipeline/pipeline.py`).

Оркестратор собирается из инъектированных фейковых провайдеров
(`fake_llm`, `fake_image_provider`) и прогоняется через публичный
`async def run(state)`. Живых вызовов и чтения секретов нет.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from loguru import logger

from app.story_pipeline.errors import (
    PermanentImageGenerationError,
    PermanentLLMError,
)
from app.story_pipeline.models import (
    Character,
    ExtractedEntities,
    JudgeVerdict,
    KeyObject,
    Location,
    NarrativeArc,
    PageScene,
    PageText,
    PagesPayload,
    RevisionPayload,
    ScenesPayload,
    Stage,
    StageMetrics,
    StoryBrief,
    StoryState,
)
from app.story_pipeline.observability import log_stage, total_usage
from app.story_pipeline.pipeline import StoryPipeline
from app.story_pipeline.providers.image import GeneratedImage
from app.story_pipeline.providers.llm import Usage


def _brief() -> StoryBrief:
    """Бриф сказки, который вернёт планировщик."""

    return StoryBrief(
        logline="Плюшевый мишка идёт через лес и находит друга.",
        controlling_idea="Смелость открывает дорогу к дружбе.",
        character_arc="Мишка учится не бояться и помогает зайцу.",
        narrative_arc=NarrativeArc(
            exposition="Мишка живёт в шкафу детской.",
            rising_action="Мишка выходит в лес и слышит плач.",
            climax="Мишка спасает зайца от ручья.",
            resolution="Мишка и заяц возвращаются домой друзьями.",
        ),
    )


def _entities() -> ExtractedEntities:
    """Реестр сущностей, на который ссылаются сцены без ремонта."""

    return ExtractedEntities(
        characters=[
            Character(
                name="Мишка",
                appearance="коричневый плюшевый медведь",
                clothing="красный вязаный шарф",
                distinguishing_features="заплатка на левом ухе",
            )
        ],
        locations=[Location(name="Лес", description="сосновый лес утром")],
        key_objects=[KeyObject(name="Мяч", description="красный мяч")],
        art_style="soft watercolor storybook illustration",
    )


def _pages(num_pages: int) -> PagesPayload:
    """Ответ сценариста: ровно `num_pages` страниц с нумерацией 1..N."""

    return PagesPayload(
        pages=[
            PageText(page_num=n, text=f"Текст страницы {n}.")
            for n in range(1, num_pages + 1)
        ]
    )


def _scenes(num_pages: int) -> ScenesPayload:
    """Ответ раскадровщика: сцены строго по именам реестра сущностей."""

    return ScenesPayload(
        scenes=[
            PageScene(
                page_num=n,
                scene_description=f"Мишка в кадре на странице {n}.",
                characters_in_scene=["Мишка"],
                location="Лес",
                objects_in_scene=[],
            )
            for n in range(1, num_pages + 1)
        ]
    )


def _ok_responses(num_pages: int = 2) -> list[object]:
    """Ответы LLM для прогона, проходящего судью с первой попытки."""

    return [
        _brief(),
        _pages(num_pages),
        _entities(),
        _scenes(num_pages),
        JudgeVerdict(passed=True, issues=[], problem_pages=[]),
    ]


def _state(num_pages: int = 2) -> StoryState:
    """Входное состояние прогона."""

    return StoryState(
        theme="Приключение плюшевого мишки",
        age=6,
        moral="Смелость помогает найти друзей",
        num_pages=num_pages,
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


def _schema_calls(llm, schema: type) -> int:
    """Сколько раз фейковый LLM вызван по конкретной схеме ответа."""

    return sum(1 for call in llm.calls if call["schema"] is schema)


async def test_run_fills_state_end_to_end(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Сквозной прогон заполняет всё состояние и пишет изображения на диск."""

    image_provider = fake_image_provider()
    llm = fake_llm(_ok_responses())
    pipeline = StoryPipeline(
        llm,
        image_provider,
        story_settings(story_output_dir=str(tmp_path)),
    )
    state = _state()

    result = await pipeline.run(state)

    assert result is state
    # Текстовые артефакты всех узлов графа.
    assert state.brief is not None
    assert [page.page_num for page in state.pages] == [1, 2]
    assert state.entities is not None
    assert [scene.page_num for scene in state.scenes] == [1, 2]
    # Иллюстрации: лист персонажей + страницы в каталоге прогона.
    assert len(state.art_prompts) == 2
    assert state.character_sheet_path == str(tmp_path / "character_sheet.png")
    assert [art.page_num for art in state.images] == [1, 2]
    assert (tmp_path / "page_1.png").exists()
    assert (tmp_path / "page_2.png").exists()
    # Каталог вывода выставлен оркестратором до иллюстрирования.
    assert state.output_dir == str(tmp_path)
    # Статус успешного прогона и временные метки.
    assert state.status == "ok"
    assert state.finished_at is not None
    # Оркестратор выставляет начало прогона сам, не полагаясь на вход.
    assert state.started_at > datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert state.finished_at >= state.started_at
    # Один вызов на каждый LLM-этап без ревизий: план, сказка, извлечение,
    # раскадровка, судья.
    assert llm.call_count == 5


async def test_pipeline_passes_reference_photo_without_storing_it(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Фото доходит до этапа иллюстрирования, но не остаётся в состоянии."""

    provider = fake_image_provider()
    photo = b"toy-photo-bytes"
    pipeline = StoryPipeline(
        fake_llm(_ok_responses()),
        provider,
        story_settings(story_output_dir=str(tmp_path)),
        reference_photo=photo,
    )
    state = _state()

    await pipeline.run(state)

    # Фото ушло единственным референсом в вызов character sheet.
    assert provider.calls[0]["reference_images"] == [photo]
    # В состоянии бинарей нет: фото не сериализуется вместе с прогоном.
    dumped = state.model_dump(mode="json")
    assert photo.decode() not in str(dumped)


async def test_pipeline_applies_configured_page_aspect_ratio(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Пропорция страниц берётся из настроек, лист всегда квадратный."""

    provider = fake_image_provider()
    pipeline = StoryPipeline(
        fake_llm(_ok_responses()),
        provider,
        story_settings(
            story_output_dir=str(tmp_path),
            story_image_aspect_ratio="3:4",
        ),
    )

    await pipeline.run(_state())

    assert provider.calls[0]["aspect_ratio"] == "1:1"
    assert [call["aspect_ratio"] for call in provider.calls[1:]] == ["3:4", "3:4"]


async def test_on_stage_reports_every_stage_in_order(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Колбэк видит имена этапов прогона в каноническом порядке."""

    reported: list[str] = []

    async def on_stage(name: str) -> None:
        reported.append(name)

    pipeline = StoryPipeline(
        fake_llm(_ok_responses()),
        fake_image_provider(),
        story_settings(story_output_dir=str(tmp_path)),
        on_stage=on_stage,
    )

    await pipeline.run(_state())

    assert reported == [stage.name for stage in pipeline.stages]
    assert "quality" in reported


async def test_on_stage_completes_before_next_stage_starts(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Оркестратор дожидается колбэка: следующий этап стартует после него.

    Регрессия на исходный дефект синхронного вызова: отправитель писал бы
    «Рисую страницы…» в уже идущем фоне, и пользователь видел бы сообщение
    не вовремя. Отметка ставится после `await sleep`, то есть момент
    завершения отправки.
    """

    events: list[str] = []
    inner_llm = fake_llm(_ok_responses())

    class _RecordingLLM:
        """Прокси `LLMClient`, фиксирующий момент вызова каждой стадии LLM."""

        async def complete_json(self, **kwargs: Any) -> tuple[Any, Any]:
            events.append("llm")
            return await inner_llm.complete_json(**kwargs)

    async def on_stage(name: str) -> None:
        events.append(f"notify-start:{name}")
        await asyncio.sleep(0)
        events.append(f"notify-done:{name}")

    pipeline = StoryPipeline(
        _RecordingLLM(),  # type: ignore[arg-type]
        fake_image_provider(),
        story_settings(story_output_dir=str(tmp_path)),
        on_stage=on_stage,
    )

    await pipeline.run(_state())

    # Все шесть этапов уведомили о себе, в порядке графа.
    order = [event for event in events if event.startswith("notify-done:")]
    assert order == [
        f"notify-done:{name}"
        for name in (
            "plan",
            "script",
            "extract",
            "scene",
            "quality",
            "illustrate",
        )
    ]
    # Работа LLM каждого этапа идёт строго после завершения уведомления
    # о нём: непосредственно перед вызовом стоит `notify-done` этого этапа.
    # Без `await` в оркестраторе уведомление «наезжало» бы на этап.
    before_llm = [
        events[index - 1]
        for index, event in enumerate(events)
        if event == "llm"
    ]
    assert before_llm == [
        "notify-done:plan",
        "notify-done:script",
        "notify-done:extract",
        "notify-done:scene",
        "notify-done:quality",
    ]


async def test_run_without_callback_is_unchanged(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Без колбэка прогон идёт как раньше: `on_stage` не передан."""

    pipeline = StoryPipeline(
        fake_llm(_ok_responses()),
        fake_image_provider(),
        story_settings(story_output_dir=str(tmp_path)),
    )
    state = _state()

    result = await pipeline.run(state)

    # Прогон без колбэка даёт тот же результат, что и раньше.
    assert result is state
    assert state.status == "ok"
    assert [art.page_num for art in state.images] == [1, 2]


async def test_run_keeps_output_dir_given_in_state(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Явно заданный каталог прогона не перетирается значением настроек.

    Регрессия на исходный дефект: `run()` безусловно писал
    `state.output_dir = story_output_dir`, из-за чего два параллельных
    прогона складывали `page_1.png` в один каталог, а лист персонажей мог
    перезаписаться между генерацией листа и страниц.
    """

    run_dir = tmp_path / "run-42"
    pipeline = StoryPipeline(
        fake_llm(_ok_responses()),
        fake_image_provider(),
        story_settings(story_output_dir=str(tmp_path / "default")),
    )
    state = _state()
    state.output_dir = str(run_dir)

    await pipeline.run(state)

    assert state.output_dir == str(run_dir)
    assert (run_dir / "page_1.png").is_file()
    assert not (tmp_path / "default").exists()


async def test_two_runs_with_own_directories_do_not_clobber(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Два прогона с разными каталогами пишут файлы независимо друг от друга."""

    first_dir = tmp_path / "user-1-1700000000"
    second_dir = tmp_path / "user-2-1700000001"

    # Разный контент у прогонов: иначе перетирание файлов не отличить
    # от корректного прогона, потому что имена у обоих наборов одинаковы.
    first_provider = fake_image_provider(
        responses=[
            GeneratedImage(data=b"sheet-one", seed=1),
            GeneratedImage(data=b"page-one", seed=2),
            GeneratedImage(data=b"page-two", seed=3),
        ]
    )
    second_provider = fake_image_provider(
        responses=[
            GeneratedImage(data=b"sheet-two", seed=11),
            GeneratedImage(data=b"page-three", seed=12),
            GeneratedImage(data=b"page-four", seed=13),
        ]
    )
    first = StoryPipeline(
        fake_llm(_ok_responses()),
        first_provider,
        story_settings(story_output_dir=str(first_dir)),
    )
    second = StoryPipeline(
        fake_llm(_ok_responses()),
        second_provider,
        story_settings(story_output_dir=str(second_dir)),
    )
    first_state = _state()
    second_state = _state()

    await first.run(first_state)
    await second.run(second_state)

    assert first_state.output_dir == str(first_dir)
    assert second_state.output_dir == str(second_dir)
    # Каждый каталог содержит ровно файлы своего прогона: оба набора
    # одинаковы по именам, поэтому перетирание видно только по содержимому.
    expected_names = ["character_sheet.png", "page_1.png", "page_2.png"]
    assert sorted(path.name for path in first_dir.iterdir()) == expected_names
    assert sorted(path.name for path in second_dir.iterdir()) == expected_names
    for run_dir, sheet_bytes, page_bytes, page2_bytes in (
        (first_dir, b"sheet-one", b"page-one", b"page-two"),
        (second_dir, b"sheet-two", b"page-three", b"page-four"),
    ):
        # Каждая картинка принадлежит своему прогону по содержимому.
        assert (run_dir / "page_1.png").read_bytes() == page_bytes
        assert (run_dir / "page_2.png").read_bytes() == page2_bytes
        assert (run_dir / "character_sheet.png").read_bytes() == sheet_bytes
    assert [art.path for art in first_state.images] == [
        str(first_dir / "page_1.png"),
        str(first_dir / "page_2.png"),
    ]


async def test_output_dir_falls_back_to_settings_when_not_set(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Каталог из настроек применяется, только если он не задан явно."""

    pipeline = StoryPipeline(
        fake_llm(_ok_responses()),
        fake_image_provider(),
        story_settings(story_output_dir=str(tmp_path)),
    )
    state = _state()
    assert state.output_dir is None

    await pipeline.run(state)

    assert state.output_dir == str(tmp_path)
    assert (tmp_path / "page_1.png").is_file()


async def test_metrics_cover_every_stage_of_the_run(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """По каждому пройденному узлу графа есть запись `StageMetrics`."""

    pipeline = StoryPipeline(
        fake_llm(_ok_responses()),
        fake_image_provider(),
        story_settings(story_output_dir=str(tmp_path)),
    )
    state = _state()

    await pipeline.run(state)

    assert [metric.stage for metric in state.metrics] == [
        "plan",
        "script",
        "extract",
        "scene",
        "judge",
        "illustrate",
    ]
    # LLM-этапы несут расход токенов, этап иллюстрирования — нет.
    for metric in state.metrics[:-1]:
        assert metric.prompt_tokens == 11
        assert metric.completion_tokens == 22
    assert state.metrics[-1].prompt_tokens is None


async def test_settings_max_revisions_bounds_revision_budget(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Оркестратор передаёт лимит ревизий из настроек, а не берёт дефолт."""

    failed = JudgeVerdict(
        passed=False,
        issues=["Сцена страницы 1 не совпадает с текстом."],
        problem_pages=[1],
    )
    revision = RevisionPayload(
        pages=[PageText(page_num=1, text="Правка текста страницы 1.")],
        scenes=[
            PageScene(
                page_num=1,
                scene_description="Правка сцены страницы 1.",
                characters_in_scene=["Мишка"],
                location="Лес",
                objects_in_scene=[],
            )
        ],
    )
    llm = fake_llm(
        [
            _brief(),
            _pages(2),
            _entities(),
            _scenes(2),
            failed,
            revision,
            failed,
        ]
    )
    # Лимит 1: судья вызывается дважды, ревизор — один раз, затем стоп.
    pipeline = StoryPipeline(
        llm,
        fake_image_provider(),
        story_settings(
            story_output_dir=str(tmp_path),
            story_max_revisions=1,
        ),
    )
    state = _state()

    await pipeline.run(state)

    assert state.revision_count == 1
    assert _schema_calls(llm, JudgeVerdict) == 2
    assert _schema_calls(llm, RevisionPayload) == 1
    assert state.status == "needs_review"
    # Изображения всё равно сгенерированы: результат прогона не потерян.
    assert [art.page_num for art in state.images] == [1, 2]


async def test_logs_cover_every_stage_without_secrets(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Лог содержит имя, время и токены каждого узла, но не секреты."""

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="INFO",
        format="{message}",
    )
    try:
        pipeline = StoryPipeline(
            fake_llm(_ok_responses()),
            fake_image_provider(),
            story_settings(
                story_output_dir=str(tmp_path),
                deepseek_api_key="SECRET-DEEPSEEK-KEY",
                gemini_api_key="SECRET-GEMINI-KEY",
            ),
        )
        await pipeline.run(_state())
    finally:
        logger.remove(sink_id)

    joined = "\n".join(messages)
    stages = ("plan", "script", "extract", "scene", "quality", "illustrate")
    for stage in stages:
        assert f"stage={stage}" in joined
    assert "duration_ms=" in joined
    assert "tokens=" in joined
    # Ключи провайдеров в лог не попадают.
    assert "SECRET-DEEPSEEK-KEY" not in joined
    assert "SECRET-GEMINI-KEY" not in joined


def test_log_stage_formats_duration_and_tokens() -> None:
    """Хелпер пишет имя этапа, вход/выход, время и токены одной строкой."""

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="INFO",
        format="{message}",
    )
    try:
        log_stage(
            "plan",
            "brief=no",
            "brief=yes",
            12.5,
            Usage(prompt_tokens=3, completion_tokens=4),
        )
    finally:
        logger.remove(sink_id)

    assert len(messages) == 1
    assert "stage=plan" in messages[0]
    assert "in=[brief=no]" in messages[0]
    assert "out=[brief=yes]" in messages[0]
    assert "duration_ms=12.5" in messages[0]
    assert "tokens=3+4" in messages[0]


def test_total_usage_sums_stage_metrics_ignoring_unknowns() -> None:
    """Суммарный расход складывает известные токены, не теряя `None`."""

    metrics = [
        StageMetrics(
            stage="judge",
            duration_ms=1.0,
            prompt_tokens=10,
            completion_tokens=2,
        ),
        StageMetrics(
            stage="reviser",
            duration_ms=1.0,
            prompt_tokens=5,
            completion_tokens=None,
        ),
    ]

    assert total_usage(metrics) == Usage(prompt_tokens=15, completion_tokens=2)
    # Если токенов нет ни у одного этапа, расход неизвестен.
    empty = StageMetrics(stage="illustrate", duration_ms=1.0)
    assert total_usage([empty]) is None


async def test_permanent_llm_error_marks_failed_and_reraises(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Постоянный сбой LLM помечает прогон failed и всплывает типом."""

    llm = fake_llm([PermanentLLMError("нет ключа DeepSeek")])
    pipeline = StoryPipeline(
        llm,
        fake_image_provider(),
        story_settings(story_output_dir=str(tmp_path)),
    )
    state = _state()

    with pytest.raises(PermanentLLMError):
        await pipeline.run(state)

    # Типизированная ошибка не подменяется, статус отражает провал прогона.
    assert state.status == "failed"
    assert state.finished_at is not None
    # Прогон остановлен: последующие этапы не выполнялись.
    assert llm.call_count == 1
    assert state.brief is None


async def test_permanent_image_error_marks_failed_and_reraises(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Постоянный сбой графического провайдера тоже помечает failed."""

    provider = fake_image_provider(
        responses=[PermanentImageGenerationError("невалидный ключ")]
    )
    pipeline = StoryPipeline(
        fake_llm(_ok_responses()),
        provider,
        story_settings(story_output_dir=str(tmp_path)),
    )
    state = _state()

    with pytest.raises(PermanentImageGenerationError):
        await pipeline.run(state)

    assert state.status == "failed"
    assert state.finished_at is not None


class _ClosableProvider:
    """Провайдер, поддерживающий закрытие: запоминает вызов `aclose`."""

    def __init__(self, *, fail: bool = False) -> None:
        self.closed = False
        self._fail = fail

    async def aclose(self) -> None:
        self.closed = True
        if self._fail:
            raise RuntimeError("сбой закрытия")


async def test_aclose_does_not_close_borrowed_providers(
    story_settings,
) -> None:
    """Инъектированные провайдеры — собственность вызывающего."""

    llm = _ClosableProvider()
    image_provider = _ClosableProvider()
    pipeline = StoryPipeline(
        llm,  # type: ignore[arg-type]
        image_provider,  # type: ignore[arg-type]
        story_settings(story_output_dir="output"),
    )

    await pipeline.aclose()

    # Провайдеры не создавались оркестратором — он не имеет права их закрывать.
    assert llm.closed is False
    assert image_provider.closed is False


async def test_aclose_closes_owned_providers_with_fault_isolation(
    story_settings,
) -> None:
    """С `owns_providers=True` закрываются оба, сбой не мешает второму."""

    llm = _ClosableProvider(fail=True)
    image_provider = _ClosableProvider()
    pipeline = StoryPipeline(
        llm,  # type: ignore[arg-type]
        image_provider,  # type: ignore[arg-type]
        story_settings(story_output_dir="output"),
        owns_providers=True,
    )

    with pytest.raises(RuntimeError):
        await pipeline.aclose()

    # Сбой закрытия первого провайдера не оставил второй незакрытым.
    assert llm.closed is True
    assert image_provider.closed is True


async def test_aclose_is_safe_for_providers_without_aclose(
    story_settings,
    fake_llm,
    fake_image_provider,
) -> None:
    """Отсутствие `aclose` у mock-провайдера не ломает закрытие."""

    pipeline = StoryPipeline(
        fake_llm([]),
        fake_image_provider(),
        story_settings(story_output_dir="output"),
        owns_providers=True,
    )

    # Не должно бросить: у обоих фейков нет `aclose`.
    await pipeline.aclose()


def test_stages_follow_protocol_in_graph_order(
    story_settings,
    fake_llm,
    fake_image_provider,
) -> None:
    """Узлы графа идут в каноническом порядке и соответствуют Stage."""

    pipeline = StoryPipeline(
        fake_llm([]),
        fake_image_provider(),
        story_settings(story_output_dir="output"),
    )

    assert [stage.name for stage in pipeline.stages] == [
        "plan",
        "script",
        "extract",
        "scene",
        "quality",
        "illustrate",
    ]
    assert all(isinstance(stage, Stage) for stage in pipeline.stages)


async def test_run_with_fakes_does_not_construct_network_clients(
    tmp_path: Path,
    monkeypatch,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """С фейковыми провайдерами оркестратор не создаёт HTTP-клиентов."""

    def _forbid_client(*args: object, **kwargs: object) -> object:
        raise AssertionError("HTTP-клиент не должен создаваться")

    monkeypatch.setattr(httpx, "AsyncClient", _forbid_client)
    pipeline = StoryPipeline(
        fake_llm(_ok_responses()),
        fake_image_provider(),
        story_settings(
            story_output_dir=str(tmp_path),
            deepseek_api_key="SECRET-DEEPSEEK-KEY",
            gemini_api_key="SECRET-GEMINI-KEY",
        ),
    )
    state = _state()

    await pipeline.run(state)

    assert state.status == "ok"
    assert [art.page_num for art in state.images] == [1, 2]
