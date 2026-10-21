"""Операции над аудио озвучки: склейка WAV, длительность и mp3.

Склейка и измерение — стандартная библиотека: провайдер отдаёт WAV одного
формата, поэтому конкатенация PCM-кадров корректна и не требует внешних
зависимостей. Склейка нужна только fallback-нарезке длинного текста и не
добавляет пауз: их расставляет сама речь. Кодирование в mp3 — единственное
место, где нужен `ffmpeg`: MAX надёжно принимает mp3/m4a, а Gemini TTS
отдаёт только PCM. Отсутствие `ffmpeg` не падает исключением, а возвращает
`None`: прогон деградирует до WAV-файла, а не теряет книжку.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from collections.abc import Sequence

from loguru import logger

# Битрейт mp3 для речи: 64 кбит/с моно достаточно и не раздувает файл.
MP3_BITRATE = "64k"


def concat_wav(segments: Sequence[bytes]) -> bytes:
    """Склеить WAV-сегменты в один непрерывный поток.

    Сегменты обязаны иметь одинаковые параметры потока: их пишет один
    провайдер одним способом. Расхождение — ошибка вызывающего кода, а не
    повод молча ресемплировать: подмена формата исказила бы голос.
    """

    if not segments:
        raise ValueError("нет сегментов для склейки")

    with wave.open(io.BytesIO(segments[0]), "rb") as first:
        channels = first.getnchannels()
        width = first.getsampwidth()
        rate = first.getframerate()
        frames: list[bytes] = [first.readframes(first.getnframes())]

    for raw in segments[1:]:
        with wave.open(io.BytesIO(raw), "rb") as segment:
            if (
                segment.getnchannels() != channels
                or segment.getsampwidth() != width
                or segment.getframerate() != rate
            ):
                raise ValueError(
                    "сегменты озвучки отличаются форматом: "
                    f"{segment.getnchannels()}x{segment.getsampwidth()}x"
                    f"{segment.getframerate()} вместо "
                    f"{channels}x{width}x{rate}"
                )
            frames.append(segment.readframes(segment.getnframes()))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(b"".join(frames))
    return buffer.getvalue()


def wav_duration_ms(data: bytes) -> int:
    """Длительность WAV в миллисекундах."""

    with wave.open(io.BytesIO(data), "rb") as source:
        rate = source.getframerate()
        if rate <= 0:  # pragma: no cover - защита от битого заголовка
            raise ValueError("частота дискретизации должна быть положительной")
        return int(source.getnframes() * 1000 / rate)


def transcode_to_mp3(wav: bytes) -> bytes | None:
    """Закодировать WAV в mp3; `None`, если `ffmpeg` недоступен или упал.

    `ffmpeg` читает из stdin и пишет в stdout: промежуточные файлы не нужны,
    а сбой кодека не оставляет мусора в каталоге прогона.
    """

    binary = shutil.which("ffmpeg")
    if binary is None:
        logger.warning(
            "ffmpeg не найден: озвучка останется в WAV и уйдёт файлом"
        )
        return None

    try:
        result = subprocess.run(
            [
                binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                MP3_BITRATE,
                "-f",
                "mp3",
                "pipe:1",
            ],
            input=wav,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr.decode("utf-8", "replace").strip()
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
            else str(exc)
        )
        logger.warning("ffmpeg не смог закодировать озвучку: {}", detail)
        return None

    if not result.stdout:
        logger.warning("ffmpeg вернул пустой mp3")
        return None
    return result.stdout


__all__ = [
    "MP3_BITRATE",
    "concat_wav",
    "transcode_to_mp3",
    "wav_duration_ms",
]
