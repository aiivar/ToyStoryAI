"""Контракт ошибок пайплайна сказок.

Отдельный небольшой модуль, чтобы этапы и провайдеры могли импортировать
типизированные ошибки, не подтягивая модели состояния. Иерархия делит
сбои на временные (есть смысл повторить) и постоянные.
"""

from __future__ import annotations


class StoryPipelineError(Exception):
    """Базовая ошибка пайплайна сказок."""


class LLMError(StoryPipelineError):
    """Ошибка обращения к текстовой модели."""


class TransientLLMError(LLMError):
    """Временный сбой LLM (сеть, 429, 5xx); повтор имеет смысл."""


class PermanentLLMError(LLMError):
    """Постоянный сбой LLM (неверный ключ, отказ); повтор бесполезен."""


class SchemaValidationError(LLMError):
    """Ответ LLM не бьётся с ожидаемой схемой даже после повтора."""


class PipelineValidationError(StoryPipelineError):
    """Артефакт этапа нарушает контракт пайплайна после корректирующего повтора.

    Отличается от `SchemaValidationError`: схема ответа валидна, но само
    содержимое (например, число или нумерация страниц) не соответствует
    требованиям этапа даже после перезапроса. Не лечится повторным разбором
    JSON, а означает невыполненный контракт данных.
    """


class ImageGenerationError(StoryPipelineError):
    """Ошибка генерации изображения."""


class TransientImageGenerationError(ImageGenerationError):
    """Временный сбой генерации изображения; повтор имеет смысл."""


class PermanentImageGenerationError(ImageGenerationError):
    """Постоянный сбой генерации изображения; повтор бесполезен."""


class NarrationError(StoryPipelineError):
    """Ошибка синтеза речи.

    Отдельная ветка иерархии нужна потому, что озвучка — единственный
    этап, сбой которого не делает прогон неуспешным: книжка без аудио
    остаётся книжкой, а причина фиксируется в состоянии.
    """


class TransientNarrationError(NarrationError):
    """Временный сбой синтеза речи (сеть, таймаут, 429, 5xx)."""


class PermanentNarrationError(NarrationError):
    """Постоянный сбой синтеза речи (неверный ключ, модель или голос)."""
