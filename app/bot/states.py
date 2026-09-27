"""Состояния FSM сценария «фото → стиль → действие»."""

from __future__ import annotations

from maxapi.context.state_machine import State, StatesGroup


class StoryFlow(StatesGroup):
    """Пошаговый сценарий создания сказки и видео."""

    waiting_photo = State()
    waiting_style = State()
    waiting_action = State()
    generating = State()


__all__ = ["StoryFlow"]
