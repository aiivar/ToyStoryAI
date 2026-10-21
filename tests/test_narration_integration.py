"""Живой вызов озвучки: включён только маркером `integration`.

По умолчанию не запускается (см. `pytest.ini`). Стоит копейки, но требует
реального `ROUTERAI_API_KEY` и сети, поэтому без ключа тест пропускается.
"""

from __future__ import annotations

import pytest

from app.story_pipeline.providers.tts import (
    GeminiSpeechProvider,
    pcm_to_wav,
)
from app.story_pipeline.settings import StorySettings, get_story_settings

pytestmark = pytest.mark.integration


async def test_live_gemini_narration_returns_audio() -> None:
    """Живой синтез короткой фразы даёт непустой WAV."""

    settings = get_story_settings()
    if not settings.routerai_api_key.strip():
        pytest.skip("ROUTERAI_API_KEY не задан")

    provider = GeminiSpeechProvider(settings, "female")
    try:
        audio = await provider.synthesize("Жил-был плюшевый мишка.")
    finally:
        await provider.aclose()

    assert audio.startswith(b"RIFF")
    assert len(audio) > len(pcm_to_wav(b""))
