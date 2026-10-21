#!/usr/bin/env python3
"""PreToolUse-хук: не пускает Claude к секретам и персональным данным.

- Read/Grep/Edit/Write по `.env` и `.env.*` (кроме `.env.example`) — запрещены.
- Edit/Write в `uploads/`, `output/` и файлы БД (`*.db`, `*.db-journal`, `*.db.bak-*`) — запрещены.
- Bash: команды, упоминающие `.env`, и `git add` путей с секретами/ПДн — запрещены.

Блокировка — exit 2: сообщение из stderr возвращается Claude.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import PurePath

# `.env`, `.env.local`, `.env.prod` и т.п., но не `.env.example`.
ENV_FILE_RE = re.compile(r"(?<![\w.-])\.env(?:\.(?!example(?![\w.-]))[\w-]+)?(?![\w.-])")
DB_NAME_RE = re.compile(r"\.db(?:-journal|\.bak[\w.-]*)?$")
PII_DIRS = ("uploads", "output")
SENSITIVE_GIT_RE = re.compile(
    r"(?<![\w.-])(?:uploads|output)(?:/|\b)|\.db(?:-journal|\.bak[\w.-]*)?(?![\w-])"
)
# Разрешённая команда из AGENTS.md: создание .env из шаблона.
ALLOWED_ENV_COMMANDS = re.compile(r"^\s*cp\s+\.env\.example\s+\.env\s*$")


def block(reason: str) -> None:
    print(f"guard_secrets: {reason}", file=sys.stderr)
    sys.exit(2)


def rel_parts(file_path: str, project_dir: str) -> tuple[str, ...]:
    path = PurePath(file_path)
    try:
        path = path.relative_to(project_dir)
    except ValueError:
        pass
    return path.parts


def is_env_file(name: str) -> bool:
    return bool(ENV_FILE_RE.fullmatch(name))


def check_path(tool: str, file_path: str, project_dir: str) -> None:
    parts = rel_parts(file_path, project_dir)
    if not parts:
        return
    name = parts[-1]
    if is_env_file(name):
        block(
            f"{tool} по {name} запрещён: там живые токены. "
            "Список переменных смотри в .env.example и app/config.py."
        )
    if tool in ("Edit", "Write"):
        if parts[0] in PII_DIRS:
            block(f"запись в {parts[0]}/ запрещена: там персональные фото и результаты генерации.")
        if DB_NAME_RE.search(name):
            block(f"запись в файл БД {name} запрещена: работа с хранилищем — через app/db/repository.py.")


def check_bash(command: str) -> None:
    if ALLOWED_ENV_COMMANDS.match(command):
        return
    if ENV_FILE_RE.search(command):
        block(
            "команда обращается к .env с живыми токенами — запрещено. "
            "Нужны имена переменных — смотри .env.example."
        )
    for segment in re.split(r"&&|\|\||;|\|", command):
        tokens = segment.split()
        if len(tokens) >= 2 and tokens[0] == "git" and tokens[1] == "add":
            args = " ".join(tokens[2:])
            if SENSITIVE_GIT_RE.search(args):
                block("git add файлов БД, uploads/ или output/ запрещён: там персональные данные.")


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()

    if tool == "Bash":
        check_bash(tool_input.get("command", ""))
    elif tool in ("Read", "Edit", "Write"):
        check_path(tool, tool_input.get("file_path", ""), project_dir)
    elif tool == "Grep":
        for key in ("path", "glob"):
            value = tool_input.get(key)
            if value:
                check_path(tool, value, project_dir)


if __name__ == "__main__":
    main()
