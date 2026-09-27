"""Точка входа приложения: сборка сервисов и запуск long polling."""

from __future__ import annotations

import asyncio

from maxapi import Bot, Dispatcher
from maxapi.exceptions import InvalidToken, MaxApiError, MaxConnection

from app.bot.handlers import BotServices, register_handlers
from app.config import Settings
from app.db.engine import Database
from app.db.repository import GenerationRepository
from app.services.photo_validation import PhotoValidator
from app.services.storage import PhotoStorage
from app.services.text_generator import TextGenerator
from app.services.video_provider import build_video_provider
from app.utils.logger import get_logger, setup_logging

logger = get_logger()


async def main() -> int:
    """Запустить бота. Возвращает код выхода процесса."""

    settings = Settings()
    setup_logging(settings)
    logger.info("Запускаю ToyStoryAI (БД: {})", settings.database_url)

    database = Database(settings.database_url)
    await database.create_schema()

    repository = GenerationRepository(database)
    storage = PhotoStorage(settings.uploads_dir)
    text_generator = TextGenerator(settings)
    video_provider = build_video_provider(settings)

    if not settings.kling_api_key:
        logger.warning(
            "KLING_API_KEY не задан: используется mock-видеопровайдер."
        )
    if not settings.deepseek_api_key:
        logger.warning(
            "DEEPSEEK_API_KEY не задан: генерация сказки будет недоступна."
        )

    bot = Bot(
        token=settings.max_bot_token,
        auto_check_subscriptions=False,
        # Отключаем авторазрешение chat/from_user через API: для личного
        # диалога get_chat_by_id отвечает chat.not.found и роняет батч
        # обновлений. Обработчики берут нужные поля из payload события.
        auto_requests=False,
    )
    dispatcher = Dispatcher()

    services = BotServices(
        settings=settings,
        validator=PhotoValidator(settings),
        storage=storage,
        text_generator=text_generator,
        video_provider=video_provider,
        repository=repository,
        download_photo=bot.download_bytes,
    )
    register_handlers(dispatcher, services)

    exit_code = 0
    try:
        await dispatcher.start_polling(bot)
    except InvalidToken:
        logger.error(
            "Неверный MAX_BOT_TOKEN. Проверьте значение в .env и запустите "
            "снова."
        )
        exit_code = 1
    except (MaxApiError, MaxConnection) as exc:
        logger.error("Не удалось подключиться к MAX API: {}", exc)
        exit_code = 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Остановка по запросу пользователя.")
    finally:
        await text_generator.aclose()
        aclose = getattr(video_provider, "aclose", None)
        if aclose is not None:
            await aclose()
        await database.dispose()
        logger.info("Сервисы остановлены.")

    return exit_code


def run() -> int:
    """Синхронная обёртка для запуска из командной строки."""

    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка по запросу пользователя.")
        return 0


if __name__ == "__main__":
    raise SystemExit(run())
