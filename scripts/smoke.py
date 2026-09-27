"""Проверка конфигурации и внешних API без запуска long polling.

Что делает:
- загружает `Settings` из `.env`/окружения;
- проверяет токен MAX через `get_me`;
- проверяет ключ и модель DeepSeek коротким запросом;
- сообщает режим видео (mock или реальный провайдер), не вызывая Kling.

Запуск из корня проекта:

    python scripts/smoke.py
    python scripts/smoke.py --skip-max          # только DeepSeek
    python scripts/smoke.py --skip-max --skip-deepseek   # только проверка конфигурации

Код выхода 0 — все выполненные проверки прошли, 1 — была ошибка.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Разрешаем `import app` при запуске как `python scripts/smoke.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from maxapi import Bot  # noqa: E402
from maxapi.exceptions import (  # noqa: E402
    InvalidToken,
    MaxApiError,
    MaxConnection,
)
from pydantic import ValidationError  # noqa: E402

from app.config import Settings  # noqa: E402
from app.services.video_provider import (  # noqa: E402
    MockVideoProvider,
    build_video_provider,
)

OK = "[OK]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
INFO = "[INFO]"


async def check_max(settings: Settings) -> bool:
    """Проверить токен MAX через get_me."""

    bot = Bot(token=settings.max_bot_token, auto_check_subscriptions=False)
    try:
        me = await bot.get_me()
        print(f"{OK} MAX: бот @{me.username} (id={me.user_id})")
        return True
    except InvalidToken:
        print(f"{FAIL} MAX: неверный MAX_BOT_TOKEN")
        return False
    except (MaxApiError, MaxConnection) as exc:
        print(f"{FAIL} MAX: нет связи с platform-api2.max.ru — {exc}")
        return False
    finally:
        await bot.close_session()


async def check_deepseek(settings: Settings) -> bool:
    """Проверить ключ и модель DeepSeek коротким запросом."""

    if not settings.deepseek_api_key:
        print(f"{SKIP} DeepSeek: DEEPSEEK_API_KEY не задан")
        return False

    url = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.deepseek_model,
        "messages": [
            {"role": "user", "content": "Ответь одним словом: проверка связи."}
        ],
        "max_tokens": 512,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(
            timeout=settings.deepseek_timeout
        ) as client:
            response = await client.post(url, json=payload, headers=headers)
    except httpx.TransportError as exc:
        print(f"{FAIL} DeepSeek: сетевая ошибка — {exc}")
        return False

    if response.status_code == 200:
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError):
            content = "<не удалось разобрать ответ>"
        print(
            f"{OK} DeepSeek: модель {settings.deepseek_model} ответила "
            f"{content.strip()!r}"
        )
        return True

    hint = ""
    if response.status_code in (400, 404):
        hint = (
            " — проверьте DEEPSEEK_MODEL: возможно, у аккаунта другая "
            "модель (например, deepseek-chat)"
        )
    print(
        f"{FAIL} DeepSeek: HTTP {response.status_code}"
        f"{hint} — {response.text[:200]}"
    )
    return False


def report_video_mode(settings: Settings) -> None:
    """Сообщить, какой видеопровайдер выбран, без реального вызова."""

    provider = build_video_provider(settings)
    if isinstance(provider, MockVideoProvider):
        print(
            f"{SKIP} Видео: KLING_API_KEY пуст → mock-провайдер, "
            f"видео будет ссылкой {settings.mock_video_url}"
        )
    else:
        print(
            f"{INFO} Видео: KLING_API_KEY задан → реальный провайдер "
            f"({settings.kling_model} @ {settings.kling_base_url}); "
            "реальный вызов в этом режиме не выполняется"
        )


def load_settings() -> Settings | None:
    """Загрузить настройки с понятным сообщением об ошибке."""

    try:
        return Settings()
    except ValidationError as exc:
        print(
            f"{FAIL} Не удалось загрузить настройки. Скопируйте "
            ".env.example в .env и заполните значения."
        )
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"       - {field}: {error['msg']}")
        return None


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-проверка ToyStoryAI")
    parser.add_argument("--skip-max", action="store_true")
    parser.add_argument("--skip-deepseek", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    if settings is None:
        return 1

    print(f"{INFO} Модель сказки: {settings.deepseek_model}")
    print(f"{INFO} База: {settings.database_url}")

    checks: list[bool] = []

    if args.skip_max:
        print(f"{SKIP} MAX: проверка пропущена")
    else:
        checks.append(await check_max(settings))

    if args.skip_deepseek:
        print(f"{SKIP} DeepSeek: проверка пропущена")
    else:
        checks.append(await check_deepseek(settings))

    report_video_mode(settings)

    if not checks:
        print(f"{INFO} Выполнены только проверки конфигурации.")
        return 0

    passed = all(checks)
    print(f"{OK if passed else FAIL} Итог: "
          f"{sum(checks)}/{len(checks)} проверок пройдено")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
