"""Точка входа приложения: сборка сервисов и запуск long polling."""

from __future__ import annotations

import asyncio

from maxapi import Bot, Dispatcher
from maxapi.exceptions import InvalidToken, MaxApiError, MaxConnection

from app.bot.handlers import BotServices, register_handlers
from app.bot.messaging import ChatSender
from app.bot.runner import build_story_pipeline
from app.config import Settings
from app.db.engine import Database
from app.db.repository import GenerationRepository
from app.services.photo_validation import PhotoValidator
from app.services.storage import PhotoStorage
from app.services.toy_vision import ToyVisionService
from app.story_pipeline.settings import StorySettings
from app.utils.logger import get_logger, setup_logging

logger = get_logger()

# Размер mock-картинки: заглушка провайдера рисует квадрат этого размера.
MOCK_IMAGE_SIZE = 32


async def main() -> int:
    """Запустить бота. Возвращает код выхода процесса."""

    settings = Settings()
    setup_logging(settings)
    logger.info("Запускаю ToyStoryAI (БД: {})", settings.database_url)

    database = Database(settings.database_url)
    await database.create_schema()

    repository = GenerationRepository(database)
    storage = PhotoStorage(settings.uploads_dir)
    story_settings = StorySettings()

    if not settings.routerai_api_key:
        logger.warning(
            "ROUTERAI_API_KEY не задан: страницы будут с mock-картинками "
            f"{MOCK_IMAGE_SIZE}×{MOCK_IMAGE_SIZE}. Такой результат нельзя "
            "показывать пользователям — задайте ключ RouterAI."
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

    # Отправитель привязан к боту, а не к событию: фоновой генерации нужно
    # писать в чат спустя минуты после апдейта, когда событие уже мертво.
    sender = ChatSender(bot)
    vision = ToyVisionService(settings)

    services = BotServices(
        settings=settings,
        validator=PhotoValidator(settings),
        storage=storage,
        repository=repository,
        download_photo=bot.download_bytes,
        sender=sender,
        vision=vision,
        make_pipeline=build_story_pipeline,
        story_settings=story_settings,
    )
    # Раннер собирается при регистрации обработчиков и возвращается наружу:
    # при остановке polling именно он должен сообщить незавершённым прогонам
    # о прерывании (D17).
    handlers = register_handlers(dispatcher, services)

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
        # Уведомление и отмена идут до закрытия провайдеров: иначе
        # пользователь ждал бы молча, а провайдеры закрылись бы под ногами
        # ещё идущего платного прогона.
        await handlers.runs.shutdown()
        await vision.aclose()
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
