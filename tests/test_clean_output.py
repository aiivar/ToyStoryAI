"""Тесты скрипта очистки тестовых прогонов (`scripts/clean_output.py`).

Скрипт лежит вне пакета `app`, поэтому загружается по пути. Проверяется
главное обещание: пользовательские каталоги не удаляются ни при каких
фильтрах, а без `--yes` не удаляется вообще ничего.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clean_output.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("clean_output", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses ищут модуль в `sys.modules` при обработке аннотаций.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


clean_output = _load_module()


def make_run(root: Path, name: str, *, files: int = 1) -> Path:
    """Создать каталог прогона с файлами внутри."""

    path = root / name
    path.mkdir(parents=True)
    for index in range(files):
        (path / f"page_{index}.png").write_bytes(b"x" * 1024)
    return path


def test_parse_run_dir_reads_user_and_time(tmp_path: Path) -> None:
    """Формат имени разбирается в user_id и время старта."""

    run = clean_output.parse_run_dir(
        tmp_path / "206619495-20260102-030405-123456"
    )

    assert run is not None
    assert run.user_id == "206619495"
    assert run.started_at == datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc
    )


@pytest.mark.parametrize(
    "name",
    ["20260102-030405", "notes", "1-20260102-030405-1234567", "1-20261340-030405-123456"],
)
def test_parse_run_dir_rejects_foreign_names(tmp_path: Path, name: str) -> None:
    """Не наш формат или несуществующая дата — не прогон."""

    assert clean_output.parse_run_dir(tmp_path / name) is None


def test_dry_run_keeps_everything(tmp_path: Path, capsys) -> None:
    """Без --yes каталоги остаются, а план печатается."""

    test_run = make_run(tmp_path, "1-20260101-120000-000001")
    user_run = make_run(tmp_path, "2-20260101-120000-000001")

    code = clean_output.main(["--dir", str(tmp_path), "--user-id", "1"])

    assert code == 0
    assert test_run.is_dir() and user_run.is_dir()
    out = capsys.readouterr().out
    assert "Это план" in out
    assert "пользовательские — не трогаем" in out


def test_yes_deletes_only_test_users(tmp_path: Path, capsys) -> None:
    """--yes удаляет прогоны тестового id и не трогает чужие каталоги."""

    test_run = make_run(tmp_path, "1-20260101-120000-000001")
    user_run = make_run(tmp_path, "2-20260101-120000-000001")
    unknown = tmp_path / "notes"
    unknown.mkdir()
    (tmp_path / ".DS_Store").write_bytes(b"x")

    code = clean_output.main(
        ["--dir", str(tmp_path), "--user-id", "1", "--yes"]
    )

    assert code == 0
    assert not test_run.exists()
    assert user_run.is_dir()
    assert unknown.is_dir()
    assert (tmp_path / ".DS_Store").is_file()


def test_keep_keeps_newest_test_runs(tmp_path: Path) -> None:
    """--keep сохраняет N самых свежих тестовых прогонов."""

    old = make_run(tmp_path, "1-20260101-120000-000001")
    fresh = make_run(tmp_path, "1-20260201-120000-000001")

    code = clean_output.main(
        ["--dir", str(tmp_path), "--user-id", "1", "--keep", "1", "--yes"]
    )

    assert code == 0
    assert fresh.is_dir()
    assert not old.exists()


def test_older_than_keeps_recent_runs(tmp_path: Path) -> None:
    """--older-than не удаляет прогоны свежее указанного срока."""

    old = make_run(tmp_path, "1-20200101-120000-000001")
    fresh = make_run(tmp_path, "1-20990101-120000-000001")

    code = clean_output.main(
        [
            "--dir",
            str(tmp_path),
            "--user-id",
            "1",
            "--older-than",
            "7",
            "--yes",
        ]
    )

    assert code == 0
    assert not old.exists()
    assert fresh.is_dir()


def test_keep_does_not_protect_user_dirs_from_delete(tmp_path: Path) -> None:
    """Пользовательский каталог не удаляется даже при --keep 0."""

    user_run = make_run(tmp_path, "74623890-20260101-120000-000001")

    code = clean_output.main(["--dir", str(tmp_path), "--user-id", "1", "--yes"])

    assert code == 0
    assert user_run.is_dir()


def test_missing_directory_is_an_error(tmp_path: Path, capsys) -> None:
    """Несуществующий каталог — понятная ошибка и код 1."""

    code = clean_output.main(["--dir", str(tmp_path / "nope")])

    assert code == 1
    assert "не найден" in capsys.readouterr().err


def test_default_test_users_do_not_include_real_accounts() -> None:
    """В наборе по умолчанию только фикстурный `1`: реальные аккаунты не чистим."""

    assert clean_output.DEFAULT_TEST_USER_IDS == ("1",)
