"""Этап озвучки: текст страниц → один аудиофайл прогона.

Узел графа, который работает параллельно иллюстрированию (см. `pipeline`).
Он озвучивает **только тексты страниц** в порядке номеров: вступление,
заголовок и «Конец» не дописываются — это продуктовое требование, а не
экономия.

Весь текст сказки уходит провайдеру **одним запросом**: постраничный синтез
давал разные темп и громкость на соседних страницах и слышимые швы. Пауз
этап не добавляет — их расставляет сама речь на знаках препинания. Нарезка
остаётся только страховкой от лимита модели и идёт по границам предложений,
без привязки к страницам.

Сбой озвучки — управляемая деградация, а не исключение: книжка уже
оплачена и обязана дойти до родителя без аудио. Причина остаётся в
`narration_error` и в метрике этапа, статус прогона не меняется.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Sequence

from loguru import logger

from app.story_pipeline.audio import (
    concat_wav,
    transcode_to_mp3,
    wav_duration_ms,
)
from app.story_pipeline.models import StageMetrics, StoryState
from app.story_pipeline.providers.tts import NarrationProvider
from app.story_pipeline.settings import StorySettings

# Имя этапа в метриках и в прогрессе бота.
STAGE_NAME = "narration"

# Предел текста на один запрос синтеза. Прокси для лимита промпта модели
# (8192 токена): максимальная сказка из настроек — 12 страниц по 120 слов,
# это примерно 9–10 тыс. знаков. Нарезка срабатывает только на аномально
# длинном тексте и идёт по границам предложений.
MAX_CHUNK_CHARS = 12000

# Имена файлов озвучки в каталоге прогона.
MP3_FILENAME = "narration.mp3"
WAV_FILENAME = "narration.wav"


class NarrationStage:
    """Озвучить сказку целиком и сохранить один аудиофайл в каталог прогона."""

    name = STAGE_NAME

    def __init__(
        self,
        provider: NarrationProvider,
        settings: StorySettings,
        *,
        concurrency: int | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        # Параллелизм нужен только fallback-нарезке длинного текста:
        # обычная сказка синтезируется одним запросом.
        self._concurrency = max(
            1, concurrency or settings.routerai_narration_concurrency
        )

    async def run(self, state: StoryState) -> StoryState:
        """Синтезировать сказку и не выбросить наружу сбой синтеза."""

        started = time.perf_counter()
        attempts = 0
        try:
            text = self._text(state)
            if not text:
                raise ValueError("в состоянии нет страниц для озвучки")
            chunks = _split_text(text, MAX_CHUNK_CHARS)
            attempts = len(chunks)
            segments = await self._synthesize(chunks)
            wav = concat_wav(segments)
            mp3 = await asyncio.to_thread(transcode_to_mp3, wav)
            path = self._write(state, mp3 if mp3 is not None else wav)
            state.narration_path = str(path)
            state.narration_error = None
            logger.info(
                "narration: озвучка готова, страниц={}, сегментов={}, "
                "длительность={} мс, файл={}",
                len(state.pages),
                len(segments),
                wav_duration_ms(wav),
                path.name,
            )
        except Exception as exc:  # noqa: BLE001 - озвучка не роняет книжку
            state.narration_path = None
            state.narration_error = f"{exc.__class__.__name__}: {exc}"
            logger.error("narration: озвучка не удалась: {}", exc)
        finally:
            state.metrics.append(
                StageMetrics(
                    stage=self.name,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    attempts=max(1, attempts),
                    error=state.narration_error,
                )
            )
        return state

    # --- Внутренняя реализация -------------------------------------------

    def _text(self, state: StoryState) -> str:
        """Один текст сказки: страницы по порядку через пустую строку.

        Пустая строка — подсказка модели сделать паузу на границе страницы;
        отдельного сегмента и отдельного запроса для страницы нет.
        """

        texts = [
            page.text.strip()
            for page in sorted(state.pages, key=lambda item: item.page_num)
            if page.text.strip()
        ]
        return "\n\n".join(texts)

    async def _synthesize(self, chunks: Sequence[str]) -> list[bytes]:
        """Синтезировать чанки с ограничением параллелизма, сохранив порядок."""

        semaphore = asyncio.Semaphore(self._concurrency)

        async def one(index: int, text: str) -> tuple[int, bytes]:
            async with semaphore:
                audio = await self._provider.synthesize(text)
            return index, audio

        results = await asyncio.gather(
            *(one(index, text) for index, text in enumerate(chunks))
        )
        return [audio for _, audio in sorted(results, key=lambda item: item[0])]

    def _write(self, state: StoryState, data: bytes) -> Path:
        """Записать аудиофайл в каталог прогона рядом со страницами."""

        directory = Path(state.output_dir or self._settings.story_output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (WAV_FILENAME if data.startswith(b"RIFF") else MP3_FILENAME)
        path.write_bytes(data)
        return path


def _split_text(text: str, limit: int) -> list[str]:
    """Разбить текст на куски не длиннее лимита по границам предложений."""

    if len(text) <= limit:
        return [text]

    sentences = _sentences(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        # Одно предложение длиннее лимита: режем по словам, чтобы не
        # превысить лимит модели и не потерять текст.
        chunks.extend(_split_long(sentence, limit))
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _sentences(text: str) -> list[str]:
    """Разбить текст на предложения, сохранив знаки конца."""

    sentences: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in ".!?…":
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _split_long(sentence: str, limit: int) -> list[str]:
    """Разбить слишком длинное предложение по словам."""

    parts: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = word
    if current:
        parts.append(current)
    return parts


__all__ = [
    "MAX_CHUNK_CHARS",
    "MP3_FILENAME",
    "NarrationStage",
    "STAGE_NAME",
    "WAV_FILENAME",
]
