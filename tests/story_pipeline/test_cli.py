"""Тесты CLI пайплайна сказок (`app/story_pipeline/cli.py`).

CLI вызывается через публичную точку `run_cli(argv, ...) -> int` с
инъектированными настройками и фейковыми провайдерами: сеть и секреты не
задействованы. Проверяются разбор аргументов, валидация диапазонов,
раскладка `output/{timestamp}/`, коды возврата и владение провайдерами.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typing import Any

from app.story_pipeline import cli
from app.story_pipeline.errors import PermanentLLMError
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
    StoryBrief,
    StoryState,
)

# Детерминированный момент прогона для проверки имени каталога.
FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
FIXED_TIMESTAMP = "20260101-120000"


def _fixed_now() -> datetime:
    """Инъектируемые часы CLI: фиксированный момент в UTC."""

    return FIXED_NOW


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
    """Ответ сценариста: ровно `num_pages` страниц нужного объёма 1..N.

    Текст добивается до 70 слов: дефолтные границы объёма страницы
    применяются к каждому прогону, и короткая заготовка означала бы лишний
    корректирующий перезапрос, которого эти тесты не проверяют.
    """

    return PagesPayload(
        pages=[
            PageText(
                page_num=n,
                text=f"Текст страницы {n}. " + " ".join(["слово"] * 70),
            )
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


def _ok_responses(num_pages: int = 2) -> list[Any]:
    """Ответы LLM для прогона, проходящего судью с первой попытки."""

    return [
        _brief(),
        _pages(num_pages),
        _entities(),
        _scenes(num_pages),
        JudgeVerdict(passed=True, issues=[], problem_pages=[]),
    ]


def _needs_review_responses(num_pages: int = 2) -> list[Any]:
    """Ответы LLM для исчерпания лимита ревизий и статуса `needs_review`."""

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
    return [
        _brief(),
        _pages(num_pages),
        _entities(),
        _scenes(num_pages),
        failed,
        revision,
        failed,
    ]


def test_invalid_age_is_rejected_before_generation(
    tmp_path: Path,
    capsys,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Возраст вне `STORY_AGE_MIN..STORY_AGE_MAX` — ошибка и ненулевой код."""

    llm = fake_llm(_ok_responses())
    image = fake_image_provider()
    settings = story_settings(story_output_dir=str(tmp_path))

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "99",
            "--moral",
            "Смелость помогает найти друзей",
        ],
        settings=settings,
        llm=llm,
        image_provider=image,
        now=_fixed_now,
    )

    captured = capsys.readouterr()
    assert code != 0
    assert "возраст" in captured.err
    assert str(settings.story_age_min) in captured.err
    assert str(settings.story_age_max) in captured.err
    # Генерация не началась: провайдеры не вызывались, каталог не создан.
    assert llm.call_count == 0
    assert image.calls == []
    assert list(tmp_path.iterdir()) == []


def test_successful_run_writes_timestamped_output(
    tmp_path: Path,
    capsys,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Успешный прогон создаёт `output/{timestamp}/` с PNG и `story.json`."""

    llm = fake_llm(_ok_responses(2))
    image = fake_image_provider()
    settings = story_settings(story_output_dir=str(tmp_path))

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "2",
        ],
        settings=settings,
        llm=llm,
        image_provider=image,
        now=_fixed_now,
    )

    captured = capsys.readouterr()
    run_dir = tmp_path / FIXED_TIMESTAMP
    assert code == 0
    assert run_dir.is_dir()
    # PNG-файлы: лист персонажей и по одному на страницу.
    assert (run_dir / "character_sheet.png").is_file()
    assert (run_dir / "page_1.png").is_file()
    assert (run_dir / "page_2.png").is_file()
    # story.json — полное состояние, читаемое обратно pydantic-моделью.
    payload = json.loads((run_dir / "story.json").read_text("utf-8"))
    restored = StoryState.model_validate(payload)
    assert restored.theme == "Приключение мишки"
    assert restored.age == 5
    assert restored.moral == "Смелость помогает найти друзей"
    assert restored.num_pages == 2
    assert restored.status == "ok"
    assert restored.output_dir == str(run_dir)
    assert restored.character_sheet_path == str(
        run_dir / "character_sheet.png"
    )
    assert [art.page_num for art in restored.images] == [1, 2]
    # Пользователю печатается итоговый статус и путь к каталогу.
    assert "ok" in captured.out
    assert FIXED_TIMESTAMP in captured.out


def test_permanent_error_returns_failure_without_traceback(
    tmp_path: Path,
    capsys,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Постоянная ошибка прогона → код 1 и понятное сообщение, без трассировки."""

    llm = fake_llm([PermanentLLMError("нет ключа DeepSeek")])
    image = fake_image_provider()
    settings = story_settings(story_output_dir=str(tmp_path))

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
        ],
        settings=settings,
        llm=llm,
        image_provider=image,
        now=_fixed_now,
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "нет ключа DeepSeek" in captured.err
    assert "Traceback" not in captured.err
    # Прогон не завершён — каталог прогона и полное состояние не созданы.
    assert not (tmp_path / FIXED_TIMESTAMP).exists()


def test_cli_closes_providers_it_created(
    tmp_path: Path,
    monkeypatch,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Провайдеры, созданные CLI, закрываются по завершении прогона."""

    closed = {"llm": 0, "image": 0}
    llm = fake_llm(_ok_responses(2))
    image = fake_image_provider()

    async def close_llm() -> None:
        closed["llm"] += 1

    async def close_image() -> None:
        closed["image"] += 1

    llm.aclose = close_llm
    image.aclose = close_image
    monkeypatch.setattr(cli, "DeepSeekLLMClient", lambda settings: llm)
    monkeypatch.setattr(cli, "make_image_provider", lambda settings: image)

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "2",
        ],
        settings=story_settings(story_output_dir=str(tmp_path)),
        now=_fixed_now,
    )

    assert code == 0
    assert closed == {"llm": 1, "image": 1}


def test_cli_does_not_close_injected_providers(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Инъектированные провайдеры — собственность вызывающего, CLI их не закрывает."""

    closed = {"llm": 0, "image": 0}
    llm = fake_llm(_ok_responses(2))
    image = fake_image_provider()

    async def close_llm() -> None:
        closed["llm"] += 1

    async def close_image() -> None:
        closed["image"] += 1

    llm.aclose = close_llm
    image.aclose = close_image

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "2",
        ],
        settings=story_settings(story_output_dir=str(tmp_path)),
        llm=llm,
        image_provider=image,
        now=_fixed_now,
    )

    assert code == 0
    assert closed == {"llm": 0, "image": 0}


def test_module_entrypoint_is_runnable() -> None:
    """`python -m app.story_pipeline` работает как точка входа (help)."""

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "app.story_pipeline", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--theme" in result.stdout
    assert "--pages" in result.stdout


def test_needs_review_run_exits_zero_with_artifacts(
    tmp_path: Path,
    capsys,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """`needs_review` — завершённый прогон с артефактами, поэтому код 0."""

    llm = fake_llm(_needs_review_responses(2))
    image = fake_image_provider()
    settings = story_settings(
        story_output_dir=str(tmp_path), story_max_revisions=1
    )

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "2",
        ],
        settings=settings,
        llm=llm,
        image_provider=image,
        now=_fixed_now,
    )

    captured = capsys.readouterr()
    run_dir = tmp_path / FIXED_TIMESTAMP
    assert code == 0
    assert "needs_review" in captured.out
    assert (run_dir / "page_1.png").is_file()
    payload = json.loads((run_dir / "story.json").read_text("utf-8"))
    assert payload["status"] == "needs_review"


def test_output_flag_overrides_settings_root(
    tmp_path: Path,
    capsys,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """`--output` задаёт корень результатов вместо `STORY_OUTPUT_DIR`."""

    settings_root = tmp_path / "settings_root"
    custom_root = tmp_path / "custom_root"
    llm = fake_llm(_ok_responses(2))
    image = fake_image_provider()

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "2",
            "--output",
            str(custom_root),
        ],
        settings=story_settings(story_output_dir=str(settings_root)),
        llm=llm,
        image_provider=image,
        now=_fixed_now,
    )

    assert code == 0
    assert not settings_root.exists()
    payload = json.loads(
        (custom_root / FIXED_TIMESTAMP / "story.json").read_text("utf-8")
    )
    assert payload["output_dir"] == str(custom_root / FIXED_TIMESTAMP)


def test_default_pages_come_from_settings(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Без `--pages` берётся `STORY_DEFAULT_PAGES` из настроек."""

    llm = fake_llm(_ok_responses(3))
    image = fake_image_provider()

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
        ],
        settings=story_settings(
            story_output_dir=str(tmp_path), story_default_pages=3
        ),
        llm=llm,
        image_provider=image,
        now=_fixed_now,
    )

    assert code == 0
    payload = json.loads(
        (tmp_path / FIXED_TIMESTAMP / "story.json").read_text("utf-8")
    )
    assert payload["num_pages"] == 3
    assert len(payload["pages"]) == 3


def test_invalid_pages_is_rejected_before_generation(
    tmp_path: Path,
    capsys,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Число страниц вне `STORY_PAGES_MIN..STORY_PAGES_MAX` — ошибка."""

    llm = fake_llm(_ok_responses())
    image = fake_image_provider()
    settings = story_settings(story_output_dir=str(tmp_path))

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "100",
        ],
        settings=settings,
        llm=llm,
        image_provider=image,
        now=_fixed_now,
    )

    captured = capsys.readouterr()
    assert code != 0
    assert "страниц" in captured.err
    assert str(settings.story_pages_min) in captured.err
    assert str(settings.story_pages_max) in captured.err
    assert llm.call_count == 0
    assert image.calls == []
    assert list(tmp_path.iterdir()) == []


class _FakeNarration:
    """Провайдер озвучки для CLI: короткая тишина без сети."""

    def __init__(self) -> None:
        self.closed = 0

    async def synthesize(self, text: str) -> bytes:
        from app.story_pipeline.providers.tts import pcm_to_wav

        return pcm_to_wav(b"\x00\x00" * 2400)

    async def aclose(self) -> None:
        self.closed += 1


def test_voice_flag_defaults_to_female(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Без флага `--voice` прогон озвучивается женским голосом."""

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "1",
        ],
        settings=story_settings(story_output_dir=str(tmp_path)),
        llm=fake_llm(_ok_responses(1)),
        image_provider=fake_image_provider(),
        narration_provider=_FakeNarration(),
        now=_fixed_now,
    )

    assert code == 0
    story = json.loads((tmp_path / FIXED_TIMESTAMP / "story.json").read_text())
    assert story["voice"] == "female"


def test_voice_flag_reaches_story_and_artifact(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Явный `--voice male` попадает в состояние и даёт файл озвучки."""

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "1",
            "--voice",
            "male",
        ],
        settings=story_settings(story_output_dir=str(tmp_path)),
        llm=fake_llm(_ok_responses(1)),
        image_provider=fake_image_provider(),
        narration_provider=_FakeNarration(),
        now=_fixed_now,
    )

    assert code == 0
    run_dir = tmp_path / FIXED_TIMESTAMP
    story = json.loads((run_dir / "story.json").read_text())
    assert story["voice"] == "male"
    assert story["narration_path"]
    assert Path(story["narration_path"]).is_file()
    assert story["narration_path"].startswith(str(run_dir))


def test_narrator_flag_reaches_story_and_artifact(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Явный `--narrator` переопределяет настройку и попадает в артефакт."""

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "1",
            "--narrator",
            "third_person_limited",
        ],
        settings=story_settings(
            story_output_dir=str(tmp_path),
            story_narrator="first_person",
        ),
        llm=fake_llm(_ok_responses(1)),
        image_provider=fake_image_provider(),
        now=_fixed_now,
    )

    assert code == 0
    story = json.loads((tmp_path / FIXED_TIMESTAMP / "story.json").read_text())
    assert story["narrator"] == "third_person_limited"
    assert story["style_profile_id"] == "junior"


def test_narrator_defaults_to_settings_value(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Без флага рассказчик берётся из `STORY_NARRATOR`."""

    code = cli.run_cli(
        [
            "--theme",
            "Приключение мишки",
            "--age",
            "5",
            "--moral",
            "Смелость помогает найти друзей",
            "--pages",
            "1",
        ],
        settings=story_settings(
            story_output_dir=str(tmp_path),
            story_narrator="third_person_limited",
        ),
        llm=fake_llm(_ok_responses(1)),
        image_provider=fake_image_provider(),
        now=_fixed_now,
    )

    assert code == 0
    story = json.loads((tmp_path / FIXED_TIMESTAMP / "story.json").read_text())
    assert story["narrator"] == "third_person_limited"


def test_unknown_narrator_flag_is_rejected_before_generation(
    tmp_path: Path,
    fake_llm,
    fake_image_provider,
    story_settings,
) -> None:
    """Опечатка в `--narrator` — ошибка разбора аргументов, а не прогон."""

    llm = fake_llm(_ok_responses(1))

    with pytest.raises(SystemExit) as failure:
        cli.run_cli(
            [
                "--theme",
                "Приключение мишки",
                "--age",
                "5",
                "--moral",
                "Смелость помогает найти друзей",
                "--pages",
                "1",
                "--narrator",
                "child",
            ],
            settings=story_settings(story_output_dir=str(tmp_path)),
            llm=llm,
            image_provider=fake_image_provider(),
            now=_fixed_now,
        )

    assert failure.value.code == 2
    assert llm.call_count == 0
