"""Тесты mock-провайдера и фабрики выбора (`providers/image.py`).

Реальный провайдер RouterAI проверяется отдельно, на подменённом
`httpx.MockTransport` (см. `test_routerai_image_provider.py`): общие для
провайдеров вещи — ретраи, классификация ошибок, разбор ответа — покрыты
там. Здесь остаётся то, что свойственно только mock-провайдеру: детерминизм
по seed, валидный PNG, отсутствие сети и выбор провайдера по ключу.
"""

from __future__ import annotations

from app.story_pipeline.providers.image import (
    MockImageProvider,
    RouterAIImageProvider,
    make_image_provider,
)
from app.story_pipeline.settings import StorySettings

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _settings(**overrides: object) -> StorySettings:
    """Настройки пайплайна без чтения `.env`, с переопределениями."""

    values: dict[str, object] = {"_env_file": None}
    values.update(overrides)
    return StorySettings(**values)  # type: ignore[arg-type]


async def test_mock_provider_is_deterministic_for_same_seed() -> None:
    """Один и тот же seed даёт одинаковые байты, разные — разные."""

    provider = MockImageProvider()

    first = await provider.generate("промпт", seed=42)
    second = await provider.generate("промпт", seed=42)
    other = await provider.generate("промпт", seed=43)

    assert first.data == second.data
    assert first.data != other.data


async def test_mock_provider_returns_valid_png() -> None:
    """Mock отдаёт валидный PNG и прокидывает seed без изменений."""

    provider = MockImageProvider()

    image = await provider.generate("промпт", seed=7)

    assert image.data.startswith(PNG_SIGNATURE)
    assert image.seed == 7


async def test_mock_provider_has_no_http_client() -> None:
    """Mock не держит HTTP-клиент и работает офлайн."""

    provider = MockImageProvider()

    assert not hasattr(provider, "_client")

    image = await provider.generate("промпт", seed=1)
    assert image.data.startswith(PNG_SIGNATURE)


def test_factory_without_key_returns_mock() -> None:
    """Пустой ключ RouterAI включает mock-провайдер без HTTP-клиента."""

    provider = make_image_provider(_settings(routerai_api_key=""))

    assert isinstance(provider, MockImageProvider)
    # Mock не создаёт клиент: у него нет ни сети, ни состояния соединения.
    assert not hasattr(provider, "_client")


def test_factory_with_key_returns_routerai() -> None:
    """Непустой ключ RouterAI включает реальный провайдер."""

    provider = make_image_provider(_settings(routerai_api_key="secret"))

    assert isinstance(provider, RouterAIImageProvider)
