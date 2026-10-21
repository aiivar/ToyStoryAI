"""Этап озвучки: текст страниц → один аудиофайл прогона.

Узел графа, который работает параллельно иллюстрированию (см. `pipeline`).
Он озвучивает **только тексты страниц** в порядке номеров: вступление,
заголовок и «Конец» не дописываются — это продуктовое требование, а не
экономия. Между страницами вставляется пауза, чтобы границы были слышны.

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
    PAGE_PAUSE_MS,
    concat_wav,
    transcode_to_mp3,
    wav_duration_ms,
)
from app.story_pipeline.models import StageMetrics, StoryState
from app.story_pipeline.providers.tts import NarrationProvider
from app.story_pipeline.settings import StorySettings

# Имя этапа в метриках и в прогрессе бота.
STAGE_NAME = "narration"

# Предел текста на один запрос синтеза. Страница обычно короче, но длинная
# режется по границам предложений: так лимит модели не превращается в
# обрезанную озвучку в середине слова.
MAX_CHUNK_CHARS = 1200

# Имена файлов озвучки в каталоге прогона.
MP3_FILENAME = "narration.mp3"
WAV_FILENAME = "narration.wav"


class NarrationStage:
    """Озвучить страницы и сохранить один аудиофайл в каталог прогона."""

    name = STAGE_NAME

    def __init__(
        self,
        provider: NarrationProvider,
        settings: StorySettings,
        *,
        concurrency: int | None = None,
        pause_ms: int = PAGE_PAUSE_MS,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._concurrency = max(
            1, concurrency or settings.routerai_narration_concurrency
        )
        self._pause_ms = pause_ms

    async def run(self, state: StoryState) -> StoryState:
        """Синтезировать сказку и не выбросить наружу сбой синтеза."""

        started = time.perf_counter()
        attempts = 0
        try:
            chunks = self._chunks(state)
            if not chunks:
                raise ValueError("в состоянии нет страниц для озвучки")
            attempts = len(chunks)
            segments = await self._synthesize(chunks)
            wav = concat_wav(segments, pause_ms=self._pause_ms)
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

    def _chunks(self, state: StoryState) -> list[str]:
        """Тексты страниц в порядке номеров, нарезанные по лимиту.

        Нарезка нужна ради лимита промпта модели: страница — обычный
        случай, но длинная страница не должна обрываться на середине.
        """

        chunks: list[str] = []
        for page in sorted(state.pages, key=lambda item: item.page_num):
            text = page.text.strip()
            if not text:
                continue
            chunks.extend(_split_text(text, MAX_CHUNK_CHARS))
        return chunks

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
