"""Отправитель сообщений, привязанный к боту, а не к событию апдейта.

Событие MAX живёт один апдейт: после возврата обработчика его объект и
вложенное в него сообщение уже нельзя использовать для отправки. Фоновая
задача генерации живёт минуты, поэтому отправлять из неё можно только через
бота и явно переданный `chat_id` — ровно как скачивание фото делается через
`bot.download_bytes`, а не через объект события.

Модуль тонкий: он знает только про `send_message` бота и ничего не знает ни
о диалоге, ни о сказке. Имена и MIME-типы вложений определяет вызывающий
код (см. `app/bot/runner.py`), потому что тип файла страницы приходит от
провайдера изображений.
"""

from __future__ import annotations

from typing import Any

from maxapi import Bot

# Вложение, которое бот умеет принять: буфер файла загружается на серверы
# MAX самим `Bot.send_message` и превращается в токен вложения.
Attachment = Any


class ChatSender:
    """Отправка сообщений в чат по идентификатору через объект `Bot`."""

    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send(
        self,
        chat_id: int,
        text: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> Any:
        """Отправить сообщение в чат; вернуть ответ API, если он есть.

        `chat_id` передаётся явно, а не берётся из события: фоновой задаче
        событие уже не принадлежит.
        """

        return await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            attachments=attachments,
        )


__all__ = ["Attachment", "ChatSender"]
