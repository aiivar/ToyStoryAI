"""Тесты озвучки: провайдер RouterAI, mock, склейка WAV и этап синтеза.

Внешних вызовов нет: HTTP подменяется `httpx.MockTransport`, провайдер этапа
— локальным фейком. Пути ошибок (таймаут, не-200, пустой поток, отсутствие
`ffmpeg`) проверяются наравне с успешным путём: сбой озвучки не должен
ломать книжку.
"""

from __future__ import annotations

import asyncio
import io
import json
import shutil
import wave
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from tenacity import wait_none

from app.story_pipeline.audio import (
    concat_wav,
    transcode_to_mp3,
    wav_duration_ms,
)
from app.story_pipeline.errors import (
    PermanentNarrationError,
    TransientNarrationError,
)
from app.story_pipeline.models import PageText, StoryState
from app.story_pipeline.providers.tts import (
    SAMPLE_RATE,
    GeminiSpeechProvider,
    MockNarrationProvider,
    make_narration_provider,
    pcm_to_wav,
)
from app.story_pipeline.settings import StorySettings
from app.story_pipeline.stages import narration as narration_module
from app.story_pipeline.stages.narration import (
    MAX_CHUNK_CHARS,
    NarrationStage,
)


def settings(**overrides: object) -> StorySettings:
    """Настройки пайплайна без чтения реального `.env`."""

    return StorySettings(
        _env_file=None,  # type: ignore[call-arg]
        **overrides,
    )


def wav_of(seconds: float, *, value: int = 0) -> bytes:
    """WAV заданной длительности с постоянным уровнем."""

    frames = int(SAMPLE_RATE * seconds)
    return pcm_to_wav((value.to_bytes(2, "little", signed=True)) * frames)


class FakeNarrationProvider:
    """Провайдер этапа: пишет тексты и отдаёт WAV или заданную ошибку."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
        seconds: float = 0.1,
    ) -> None:
        self._error = error
        self._seconds = seconds
        self.texts: list[str] = []
        self.closed = 0
        self.active = 0
        self.max_active = 0

    async def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            if self._error is not None:
                raise self._error
            return wav_of(self._seconds)
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.closed += 1


def state_with_pages(tmp_path: Path, texts: list[str]) -> StoryState:
    """Состояние прогона с готовыми текстами страниц."""

    return StoryState(
        theme="Тема",
        age=5,
        moral="помогать",
        num_pages=len(texts),
        pages=[
            PageText(page_num=index + 1, text=text)
            for index, text in enumerate(texts)
        ],
        output_dir=str(tmp_path),
        started_at=datetime.now(timezone.utc),
    )


def pages_of_sentences(count: int, sentences_per_page: int) -> list[str]:
    """Страницы предсказуемого объёма: текст точно превышает лимит запроса."""

    return [
        " ".join(
            f"Страница {page}, предложение {index}."
            for index in range(sentences_per_page)
        )
        for page in range(1, count + 1)
    ]


# --- Провайдер ---------------------------------------------------------------


def test_factory_uses_mock_without_key() -> None:
    """Пустой ключ включает mock: платных вызовов нет."""

    provider = make_narration_provider(settings(), "female")
    assert isinstance(provider, MockNarrationProvider)


def test_factory_uses_gemini_with_key() -> None:
    """С ключом строится провайдер RouterAI."""

    provider = make_narration_provider(
        settings(routerai_api_key="k"), "male"
    )
    assert isinstance(provider, GeminiSpeechProvider)


@pytest.mark.asyncio
async def test_gemini_request_shape_and_wav_output() -> None:
    """Запрос уходит в audio/speech, текст дословный, ответ — WAV."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"\x00\x00" * 2400)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiSpeechProvider(
        settings(
            routerai_api_key="secret",
            routerai_narration_style="тепло и неторопливо",
        ),
        "male",
        client=client,
        retry_wait=wait_none(),
    )

    wav = await provider.synthesize("Жил-был мишка.")
    await provider.aclose()

    assert wav.startswith(b"RIFF")
    with wave.open(io.BytesIO(wav), "rb") as parsed:
        assert parsed.getnchannels() == 1
        assert parsed.getframerate() == SAMPLE_RATE

    request = seen[0]
    assert request.url.path.endswith("/audio/speech")
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["input"] == "Жил-был мишка."
    assert body["response_format"] == "pcm"
    assert body["voice"] == "Charon"
    assert body["provider"]["options"]["google-ai-studio"][
        "speech_metadata"
    ]["style"] == "тепло и неторопливо"


@pytest.mark.asyncio
async def test_gemini_retries_transient_failure() -> None:
    """Таймаут первого вызова повторяется, второй ответ доходит."""

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectTimeout("медленно")
        return httpx.Response(200, content=b"\x00\x00" * 100)

    provider = GeminiSpeechProvider(
        settings(routerai_api_key="k"),
        "female",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_attempts=2,
        retry_wait=wait_none(),
    )

    assert await provider.synthesize("текст")
    assert calls == 2


@pytest.mark.asyncio
async def test_gemini_permanent_error_is_not_retried() -> None:
    """Отказ 400 — постоянная ошибка: повторов нет."""

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad voice"})

    provider = GeminiSpeechProvider(
        settings(routerai_api_key="k"),
        "female",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_attempts=3,
        retry_wait=wait_none(),
    )

    with pytest.raises(PermanentNarrationError):
        await provider.synthesize("текст")
    assert calls == 1


@pytest.mark.asyncio
async def test_gemini_empty_audio_is_a_failure() -> None:
    """Пустой поток не считается валидным аудио."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    provider = GeminiSpeechProvider(
        settings(routerai_api_key="k"),
        "female",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_attempts=2,
        retry_wait=wait_none(),
    )

    with pytest.raises(TransientNarrationError):
        await provider.synthesize("текст")


@pytest.mark.asyncio
async def test_mock_provider_returns_valid_wav() -> None:
    """Mock отдаёт валидный WAV без сети."""

    wav = await MockNarrationProvider().synthesize("короткий текст")
    assert wav.startswith(b"RIFF")
    assert wav_duration_ms(wav) > 0


# --- Аудио -------------------------------------------------------------------


def test_concat_joins_segments_without_pause() -> None:
    """Склейка трёх сегментов равна сумме длительностей, без пауз."""

    segments = [wav_of(0.1) for _ in range(3)]
    combined = concat_wav(segments)

    assert wav_duration_ms(combined) == pytest.approx(300, abs=15)


def test_concat_rejects_different_formats() -> None:
    """Сегменты другого формата — ошибка, а не молчаливый ресемплинг."""

    first = wav_of(0.1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(b"\x00" * 4800)

    with pytest.raises(ValueError):
        concat_wav([first, buffer.getvalue()])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg не установлен"
)
def test_transcode_produces_mp3() -> None:
    """При наличии ffmpeg WAV превращается в непустой mp3."""

    mp3 = transcode_to_mp3(wav_of(0.2))
    assert mp3 is not None
    assert len(mp3) > 0


def test_transcode_without_ffmpeg_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без ffmpeg кодирование — не исключение, а `None`."""

    monkeypatch.setattr(
        "app.story_pipeline.audio.shutil.which", lambda name: None
    )
    assert transcode_to_mp3(wav_of(0.1)) is None


# --- Этап озвучки -------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_writes_single_audio_file(tmp_path: Path) -> None:
    """Один файл на всю сказку, страницы озвучены по порядку."""

    provider = FakeNarrationProvider()
    stage = NarrationStage(provider, settings())
    state = state_with_pages(tmp_path, ["Первая.", "Вторая.", "Третья."])

    await stage.run(state)

    assert state.narration_error is None
    assert state.narration_path is not None
    path = Path(state.narration_path)
    assert path.is_file()
    assert path.parent == tmp_path
    assert path.suffix in {".mp3", ".wav"}
    metric = state.metrics[-1]
    assert metric.stage == "narration"
    assert metric.error is None


@pytest.mark.asyncio
async def test_stage_sends_whole_story_in_one_request(tmp_path: Path) -> None:
    """Весь текст сказки уходит одним запросом, а не по страницам."""

    provider = FakeNarrationProvider()
    stage = NarrationStage(provider, settings())
    state = state_with_pages(tmp_path, ["Первая.", "Вторая.", "Третья."])

    await stage.run(state)

    assert provider.texts == ["Первая.\n\nВторая.\n\nТретья."]


@pytest.mark.asyncio
async def test_stage_splits_over_limit_text_by_sentences(
    tmp_path: Path,
) -> None:
    """Длинный текст режется по предложениям, а не по страницам."""

    texts = pages_of_sentences(8, 100)
    assert sum(len(text) for text in texts) > MAX_CHUNK_CHARS
    provider = FakeNarrationProvider()
    stage = NarrationStage(provider, settings())
    state = state_with_pages(tmp_path, texts)

    await stage.run(state)

    assert 1 < len(provider.texts) < len(texts)
    assert all(len(chunk) <= MAX_CHUNK_CHARS for chunk in provider.texts)
    # Первый запрос содержит несколько страниц подряд: границы страниц
    # больше не совпадают с границами запросов.
    assert "Страница 1," in provider.texts[0]
    assert "Страница 2," in provider.texts[0]
    combined = " ".join(provider.texts)
    for page in range(1, len(texts) + 1):
        assert f"Страница {page}," in combined


@pytest.mark.asyncio
async def test_stage_fallback_chunks_have_no_artificial_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сегменты fallback-нарезки склеиваются без паузы между страницами."""

    monkeypatch.setattr(
        narration_module, "transcode_to_mp3", lambda wav: None
    )
    provider = FakeNarrationProvider(seconds=0.1)
    stage = NarrationStage(provider, settings())
    state = state_with_pages(tmp_path, pages_of_sentences(8, 100))

    await stage.run(state)

    assert len(provider.texts) > 1
    assert state.narration_path is not None
    audio = Path(state.narration_path).read_bytes()
    assert wav_duration_ms(audio) == pytest.approx(
        len(provider.texts) * 100, abs=15
    )


@pytest.mark.asyncio
async def test_stage_failure_is_degradation(tmp_path: Path) -> None:
    """Сбой синтеза не меняет статус и не выбрасывает исключение."""

    provider = FakeNarrationProvider(error=TransientNarrationError("сбой"))
    stage = NarrationStage(provider, settings())
    state = state_with_pages(tmp_path, ["Страница."])

    result = await stage.run(state)

    assert result.status == "ok"
    assert result.narration_path is None
    assert "TransientNarrationError" in (result.narration_error or "")
    assert result.metrics[-1].error


@pytest.mark.asyncio
async def test_stage_limits_concurrency(tmp_path: Path) -> None:
    """Одновременных запросов fallback-нарезки не больше предела."""

    provider = FakeNarrationProvider()
    stage = NarrationStage(provider, settings(), concurrency=2)
    state = state_with_pages(tmp_path, pages_of_sentences(40, 100))

    await stage.run(state)

    assert len(provider.texts) > 2
    assert provider.max_active == 2
