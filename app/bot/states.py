"""Состояния FSM сценария «фото → возраст → мораль → постраничная сказка»."""

from __future__ import annotations

from maxapi.context.state_machine import State, StatesGroup


class StoryFlow(StatesGroup):
    """Пошаговый сценарий создания книжки-сказки с картинками.

    `waiting_style` и `waiting_action` прежнего флоу заменены на
    `waiting_age` и `waiting_moral`: пайплайн просит возраст читателя и
    мораль, а стиль и действие игрушки в новом сюжете не участвуют.

    Отдельного состояния «генерируется» нет: прогон идёт в фоне и переживает
    апдейт, поэтому его наличие проверяет реестр задач раннера, а не FSM.
    """

    waiting_photo = State()
    waiting_age = State()
    waiting_moral = State()
    waiting_voice = State()


__all__ = ["StoryFlow"]
