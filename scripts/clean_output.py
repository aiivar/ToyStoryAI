"""Очистка тестовых прогонов сказки в каталоге результатов.

Скрипт удаляет только каталоги прогонов **тестовых** пользователей: имя
каталога имеет формат `{user_id}-{YYYYMMDD}-{HHMMSS}-{микросекунды}`, и
решение принимается по `user_id` из имени. Каталоги остальных пользователей
не удаляются никогда — они попадают в отчёт как «пользовательские» и
пропускаются, даже если подходят по возрасту или `--keep`.

По умолчанию скрипт ничего не удаляет: он показывает, что будет удалено и
сколько места освободится. Удаление — только с `--yes`.

Запуск из корня проекта:

    python scripts/clean_output.py                     # план, ничего не удаляя
    python scripts/clean_output.py --yes               # удалить тестовые прогоны
    python scripts/clean_output.py --keep 1 --yes      # оставить один свежий
    python scripts/clean_output.py --older-than 7 --yes  # только старше 7 дней
    python scripts/clean_output.py --user-id 123 --yes   # другой набор тестовых id

Код выхода 0 — план построен или удаление прошло, 1 — ошибка (нет каталога,
нечего чистить по заданным фильтрам и т. п.).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

# User_id, чьи прогоны считаются тестовыми по умолчанию.
#
# `1` — фикстурный пользователь юнит-тестов: тесты раннера собирают каталог
# прогона с настройками по умолчанию и пишут в `output/1-*`. Реальные
# пользователи MAX имеют многозначные id, поэтому `1` тестовый по построению.
#
# Реальные аккаунты (включая аккаунт разработчика) сюда не входят: чужие
# каталоги скрипт не удаляет ни при каких флагах. Свой тестовый набор
# задаётся флагом `--user-id`.
DEFAULT_TEST_USER_IDS: tuple[str, ...] = ("1",)

# Формат каталога прогона: `{user_id}-{YYYYMMDD}-{HHMMSS}-{microseconds}`.
RUN_DIR_RE = re.compile(
    r"^(?P<user_id>\d+)-(?P<date>\d{8})-(?P<time>\d{6})-(?P<micro>\d{6})$"
)


@dataclass(frozen=True)
class RunDir:
    """Каталог прогона, распознанный по имени."""

    path: Path
    user_id: str
    started_at: datetime


@dataclass
class Plan:
    """Что скрипт собирается сделать, без побочных эффектов."""

    to_delete: list[RunDir]
    kept_newest: list[RunDir]
    kept_recent: list[RunDir]
    user_dirs: list[Path]
    unknown: list[Path]

    @property
    def bytes_to_free(self) -> int:
        return sum(dir_size(item.path) for item in self.to_delete)


def parse_run_dir(path: Path) -> RunDir | None:
    """Разобрать имя каталога прогона; `None` — имя не нашего формата."""

    match = RUN_DIR_RE.match(path.name)
    if match is None:
        return None
    try:
        started = datetime.strptime(
            f"{match.group('date')}{match.group('time')}", "%Y%m%d%H%M%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        # Похоже на формат, но даты не существует: такой каталог не трогаем.
        return None
    return RunDir(path=path, user_id=match.group("user_id"), started_at=started)


def scan(root: Path) -> tuple[list[RunDir], list[Path]]:
    """Собрать прогоны и всё, что не похоже на прогон."""

    runs: list[RunDir] = []
    unknown: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_run_dir(entry)
        if parsed is None:
            unknown.append(entry)
        else:
            runs.append(parsed)
    return runs, unknown


def select_runs(
    runs: Sequence[RunDir],
    *,
    test_user_ids: Iterable[str],
    keep: int = 0,
    older_than_days: int | None = None,
    now: datetime | None = None,
) -> Plan:
    """Разделить прогоны на удаляемые и пропускаемые.

    Пользовательские прогоны не удаляются никогда: фильтры `keep` и
    `older_than_days` применяются только к прогонам тестовых `user_id`.
    """

    test_ids = {str(value) for value in test_user_ids}
    moment = now or datetime.now(timezone.utc)
    threshold = (
        moment - timedelta(days=older_than_days)
        if older_than_days is not None
        else None
    )

    user_dirs = sorted(
        (run.path for run in runs if run.user_id not in test_ids)
    )
    test_runs = [run for run in runs if run.user_id in test_ids]
    test_runs.sort(key=lambda item: item.started_at, reverse=True)

    kept_newest = test_runs[:keep] if keep > 0 else []
    candidates = test_runs[keep:] if keep > 0 else test_runs

    to_delete: list[RunDir] = []
    kept_recent: list[RunDir] = []
    for run in candidates:
        if threshold is not None and run.started_at >= threshold:
            kept_recent.append(run)
        else:
            to_delete.append(run)

    return Plan(
        to_delete=to_delete,
        kept_newest=kept_newest,
        kept_recent=kept_recent,
        user_dirs=user_dirs,
        unknown=[],
    )


def dir_size(path: Path) -> int:
    """Суммарный размер файлов каталога в байтах."""

    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def human_size(size: int) -> str:
    """Размер в человекочитаемом виде (МБ с одним знаком)."""

    megabytes = size / (1024 * 1024)
    if megabytes < 0.1 and size:
        return f"{size} Б"
    return f"{megabytes:.1f} МБ"


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов с русскими подсказками."""

    parser = argparse.ArgumentParser(
        prog="python scripts/clean_output.py",
        description=(
            "Удалить каталоги прогонов тестовых пользователей. "
            "По умолчанию показывает план и ничего не удаляет."
        ),
    )
    parser.add_argument(
        "--dir",
        default="output",
        help="Корневой каталог прогонов (по умолчанию output).",
    )
    parser.add_argument(
        "--user-id",
        action="append",
        dest="user_ids",
        default=None,
        help=(
            "Тестовый user_id, чьи прогоны можно удалять. Можно указать "
            "несколько раз; по умолчанию "
            + ", ".join(DEFAULT_TEST_USER_IDS)
        ),
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=0,
        help="Оставить N самых свежих тестовых прогонов.",
    )
    parser.add_argument(
        "--older-than",
        type=int,
        default=None,
        metavar="DAYS",
        help="Удалять только прогоны старше указанного числа дней.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Удалить по-настоящему; без флага выполняется только план.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа: вернуть код 0 при успехе и 1 при ошибке."""

    args = build_parser().parse_args(argv)
    root = Path(args.dir)
    if not root.is_dir():
        print(f"Ошибка: каталог {root} не найден.", file=sys.stderr)
        return 1
    if args.keep < 0:
        print("Ошибка: --keep не может быть отрицательным.", file=sys.stderr)
        return 1
    if args.older_than is not None and args.older_than < 0:
        print(
            "Ошибка: --older-than не может быть отрицательным.",
            file=sys.stderr,
        )
        return 1

    user_ids = tuple(args.user_ids or DEFAULT_TEST_USER_IDS)
    runs, unknown = scan(root)
    plan = select_runs(
        runs,
        test_user_ids=user_ids,
        keep=args.keep,
        older_than_days=args.older_than,
    )
    plan.unknown = unknown

    print(f"Каталог: {root}")
    print(f"Тестовые user_id: {', '.join(user_ids)}")
    print(f"Найдено прогонов: {len(runs)}")
    print(f"  к удалению: {len(plan.to_delete)}")
    for run in plan.to_delete:
        print(f"    - {run.path.name} ({human_size(dir_size(run.path))})")
    if plan.kept_newest:
        print(f"  оставлены как самые свежие (--keep {args.keep}):")
        for run in plan.kept_newest:
            print(f"    - {run.path.name}")
    if plan.kept_recent:
        print("  оставлены как более свежие, чем --older-than:")
        for run in plan.kept_recent:
            print(f"    - {run.path.name}")
    if plan.user_dirs:
        print("  пользовательские — не трогаем:")
        for path in plan.user_dirs:
            print(f"    - {path.name}")
    if plan.unknown:
        print("  не похожи на прогоны — не трогаем:")
        for path in plan.unknown:
            print(f"    - {path.name}")

    if not plan.to_delete:
        print("Нечего удалять: под фильтры не попал ни один тестовый прогон.")
        return 0

    print(f"Освободится: {human_size(plan.bytes_to_free)}")
    if not args.yes:
        print("Это план. Чтобы удалить, повторите с --yes.")
        return 0

    errors = 0
    for run in plan.to_delete:
        try:
            shutil.rmtree(run.path)
        except OSError as exc:
            errors += 1
            print(f"Не удалось удалить {run.path}: {exc}", file=sys.stderr)
    print(f"Удалено: {len(plan.to_delete) - errors} из {len(plan.to_delete)}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
