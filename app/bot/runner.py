"""Фоновая генерация сказки и постраничная выдача результата.

Модуль отвечает за всё, что происходит после нажатия кнопки морали: собирает
вход пайплайна, запускает прогон отдельной задачей, сообщает о прогрессе на
границах этапов, отдаёт страницы по одной и записывает результат в БД.
Обработчик (`app/bot/handlers.py`) только разбирает событие, переключает
состояния FSM и передаёт сюда запрос — бизнес-правил и HTTP в нём нет.

Почему задача, а не `await` в обработчике: генерация занимает 30–90 секунд,
а обработчик события держит dispatcher. Внутри задачи нельзя пользоваться
объектом события MAX (он живёт один апдейт), поэтому отправка идёт через
`ChatSender`, привязанный к боту, с явно захваченным `chat_id`.

Задачи кладутся в реестр: на задачу без сильных ссылок сборщик мусора может
сработать раньше времени, и генерация молча пропала бы на середине. Реестр
заодно защищает от второго платного прогона того же пользователя и позволяет
при остановке polling сообщить незавершённым прогонам о прерывании.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from maxapi.enums.upload_type import UploadType
from maxapi.types.input_media import InputMediaBuffer

from app.bot.keyboards import (
    build_restart_keyboard,
    build_subscription_keyboard,
)
from app.bot.messaging import ChatSender
from app.bot.story_request import (
    build_run_dir,
    build_theme,
    build_toy_description,
    join_pages_text,
)
from app.db.models import GenerationStatus
from app.db.repository import GenerationRepository
from app.services.toy_vision import ToyVisionService
from app.story_pipeline.models import StoryState
from app.story_pipeline.pipeline import StageCallback, StoryPipeline
from app.story_pipeline.providers.image import make_image_provider
from app.story_pipeline.providers.llm import DeepSeekLLMClient
from app.story_pipeline.settings import StorySettings
from app.utils.image_mime import detect_image_mime_type
from app.utils.logger import get_logger

logger = get_logger()

# Отправитель, зафиксированный на чат: `await send(text=..., attachments=...)`.
SendFunc = Callable[..., Awaitable[Any]]

# Имя этапа иллюстрирования в графе пайплайна: по нему выбирается сообщение
# «Рисую страницы…», все прочие этапы считаются текстовыми.
ILLUSTRATE_STAGE = "illustrate"

# --- Тексты сообщений (пользовательские строки на русском) --------------------

STORY_PROGRESS = "Готовлю сказку…"
ILLUSTRATION_PROGRESS = "Рисую страницы…"
STORY_READY = "Сказка готова! Читайте страницы по порядку."
FREE_LIMIT_REACHED = (
    "Вы уже получили бесплатную сказку с картинками. Чтобы сделать новую, "
    "оформите подписку."
)

# Заголовок сообщения со страницей. Нумерация — из пайплайна, а не счётчик
# доставки: если файл страницы не отправился, текст всё равно приходит с
# правильным номером, и родитель не теряет нумерацию книги.
PAGE_TITLE = "Страница"

# Сообщение вместо картинки, когда файл страницы недоступен.
PAGE_WITHOUT_IMAGE = (
    "Картинку к этой странице не получилось приложить — текст выше остаётся "
    "у вас."
)

# Страница без иллюстрации: прогон завершился, но картинки нет (этап
# иллюстрирования отметил пропуск).
PAGE_IMAGE_SKIPPED = (
    "Картинку к этой странице отрисовать не вышло. Текст страницы выше "
    "остаётся у вас."
)

# Итог частичного прогона: каких страниц не хватило.
#
# Причина в чате — фиксированная русская фраза. Технический текст из
# `StageMetrics.error` (английский, с HTTP-кодами и URL провайдера) в чат не
# попадает: родителю он ничего не объясняет, а показывать ссылку на
# эндпоинт в личном диалоге незачем. Техническая причина уходит в лог и в
# запись о прогоне — там она и нужна.
PARTIAL_TITLE = "К сожалению, сказка получилась не полной"
PARTIAL_REASON = "картинку к ней отрисовать не вышло: сервис не ответил"
PARTIAL_RESULT = (
    f"{PARTIAL_TITLE}: без картинок остались страницы {{numbers}}. "
    f"Причина: {PARTIAL_REASON}."
)

# Повторная попытка предлагается один раз: без ограничения переделки
# разошлись бы по цене.
PARTIAL_REDO = (
    "Могу попробовать ещё раз и сделать сказку заново — скажите «Начать "
    "заново». Повторять без конца не буду, это стоит денег."
)
VISION_FAILED = (
    "Не получилось разглядеть игрушку на фотографии. Пришлите фото ещё "
    "раз — лучше при дневном свете и ближе к камере."
)
GENERATION_FAILED = (
    "Не получилось создать сказку с картинками. Попробуйте начать заново."
)
RUN_INTERRUPTED = (
    "Генерация сказки прервана: бот остановлен. Попробуйте начать заново."
)


class PipelineFactory(Protocol):
    """Сборка пайплайна под конкретный прогон.

    Отдельная фабрика нужна, потому что у каждого прогона свои входные
    данные — байты фото как референс и асинхронный колбэк прогресса, — и
    своя копия настроек с уникальным каталогом вывода.
    """

    def __call__(
        self,
        *,
        settings: StorySettings,
        reference_photo: bytes | None,
        on_stage: StageCallback | None,
    ) -> StoryPipeline: ...


@dataclass(frozen=True)
class RunRequest:
    """Запрос на генерацию: всё, что обработчик узнал в диалоге.

    `photo` — байты принятой фотографии: они нужны и распознаванию
    игрушки, и пайплайну как референс character sheet. `photo_path` —
    путь к сохранённому файлу, он попадает в запись о генерации.
    """

    user_id: int
    chat_id: int
    tale_id: int
    photo_path: str
    photo: bytes
    age: int
    moral: str


@dataclass
class _ActiveRun:
    """Незавершённый прогон: задача и адресат уведомлений."""

    task: asyncio.Task[None]
    chat_id: int
    tale_id: int


def numbers_text(numbers: Sequence[int]) -> str:
    """Номера страниц через запятую для сообщения пользователю."""

    return ", ".join(str(number) for number in numbers)


def _read_page_attachment(path: str) -> InputMediaBuffer:
    """Собрать вложение страницы из файла каталога прогона.

    Имя берётся у самого файла: пайплайн называет страницы по типу, который
    вернул провайдер (`image/jpeg` → `.jpg`), и переименование положило бы в
    чат JPEG-байты под именем `.png`. Класс загрузки определяется по
    сигнатуре содержимого тем же помощником, что и у провайдера картинок:
    расширение приходит от вендора, а содержимое — факт.
    """

    file_path = Path(path)
    data = file_path.read_bytes()
    mime_type = detect_image_mime_type(data)
    upload_type = (
        UploadType.IMAGE if mime_type is not None else UploadType.FILE
    )
    return InputMediaBuffer(
        buffer=data, filename=file_path.name, type=upload_type
    )


def build_story_pipeline(
    *,
    settings: StorySettings,
    reference_photo: bytes | None = None,
    on_stage: StageCallback | None = None,
) -> StoryPipeline:
    """Собрать пайплайн с провайдерами, созданными по настройкам прогона.

    Провайдеры принадлежат пайплайну (`owns_providers=True`), поэтому
    вызывающий закрывает его сам: у каждого прогона свои HTTP-клиенты, и
    держать их после завершения прогона незачем.
    """

    return StoryPipeline(
        DeepSeekLLMClient(settings),
        make_image_provider(settings),
        settings,
        owns_providers=True,
        reference_photo=reference_photo,
        on_stage=on_stage,
    )


class StoryRunner:
    """Фоновые прогоны сказки: запуск, учёт, выдача результата."""

    def __init__(
        self,
        *,
        repository: GenerationRepository,
        vision: ToyVisionService,
        make_pipeline: PipelineFactory,
        story_settings: StorySettings,
        sender: ChatSender,
    ) -> None:
        self._repository = repository
        self._vision = vision
        self._make_pipeline = make_pipeline
        self._settings = story_settings
        self._sender = sender
        # Реестр незавершённых прогонов по пользователю. Ссылка на задачу
        # здесь и есть то, что не даёт сборщику мусора убрать её раньше
        # времени.
        self._runs: dict[int, _ActiveRun] = {}
        # Пользователи, которым уже предлагали повторить частичный прогон.
        # Предложение одно: иначе неограниченные переделки разошлись бы по
        # цене (см. D21).
        self._redo_offered: set[int] = set()

    # --- Свойства -------------------------------------------------------

    @property
    def num_pages(self) -> int:
        """Число страниц сказки по умолчанию из настроек пайплайна."""

        return self._settings.story_default_pages

    @property
    def pending(self) -> int:
        """Сколько прогонов выполняется прямо сейчас."""

        return len(self._runs)

    def is_running(self, user_id: int) -> bool:
        """Идёт ли сейчас прогон пользователя."""

        return user_id in self._runs

    # --- Запуск и остановка ---------------------------------------------

    def start(self, request: RunRequest) -> bool:
        """Запустить фоновую генерацию; `False` — прогон уже идёт.

        Метод синхронный: между проверкой и созданием задачи нет `await`,
        поэтому два события подряд (повторное нажатие кнопки или повторная
        доставка) не успеют запустить второй платный прогон.
        """

        if request.user_id in self._runs:
            logger.info(
                "Повторный запуск отброшен: прогон пользователя {} уже идёт",
                request.user_id,
            )
            return False

        task = asyncio.create_task(self._run_guarded(request))
        run = _ActiveRun(
            task=task, chat_id=request.chat_id, tale_id=request.tale_id
        )
        self._runs[request.user_id] = run
        task.add_done_callback(
            lambda finished, user_id=request.user_id: self._finish(
                user_id, finished
            )
        )
        return True

    async def wait_idle(self) -> None:
        """Дождаться завершения текущих прогонов (тесты, остановка бота)."""

        tasks = [run.task for run in self._runs.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self) -> None:
        """Сообщить незавершённым прогонам о прерывании и отменить их.

        Вызывается при остановке polling до закрытия провайдеров: иначе
        пользователь ждал бы молча, а провайдеры закрылись бы под ногами
        ещё идущего прогона.
        """

        runs = list(self._runs.values())
        self._runs.clear()
        for run in runs:
            await self._notify_interrupted(run.chat_id)
            run.task.cancel()
        if runs:
            await asyncio.gather(
                *(run.task for run in runs), return_exceptions=True
            )

    # --- Прогон ----------------------------------------------------------

    async def run(self, request: RunRequest) -> None:
        """Выполнить генерацию: квота → распознавание → пайплайн → выдача.

        Проверка квоты идёт первым шагом, до распознавания: распознавание
        игрушки — платный вызов, и запускать его пользователю, которому
        сказка уже выдана, незачем.
        """

        send = self._sender_for(request.chat_id)

        if await self._repository.has_completed_tale(request.user_id):
            logger.info(
                "Квота пользователя {} израсходована, прогон не запускаю",
                request.user_id,
            )
            # Запись о прогоне закрываем сразу: иначе она навсегда осталась бы
            # «в работе», хотя платной работы не было.
            await self._record(
                request,
                status=GenerationStatus.FAILED,
                error="квота израсходована, прогон не запускался",
            )
            await send(
                FREE_LIMIT_REACHED, [build_subscription_keyboard().as_markup()]
            )
            return

        try:
            description = await self._vision.describe(request.photo)
        except Exception as exc:  # noqa: BLE001 - сеть и LLM любят удивлять
            logger.error("Распознавание игрушки не удалось: {}", exc)
            await self._record(
                request, status=GenerationStatus.FAILED, error=str(exc)
            )
            await send(VISION_FAILED, [build_restart_keyboard().as_markup()])
            return

        # Сборка настроек, состояния и пайплайна идёт внутри `try`:
        # создание клиента LLM и провайдера картинок тоже умеет падать
        # (неверное соотношение сторон, пустой ключ), и такой сбой обязан
        # закончиться сообщением пользователю, а не тишиной после
        # «Запускаю сказку!».
        pipeline: StoryPipeline | None = None
        try:
            run_settings = self._settings.model_copy(
                update={
                    "story_output_dir": build_run_dir(
                        request.user_id, self._settings
                    )
                }
            )
            announced: set[str] = set()

            async def on_stage(stage_name: str) -> None:
                await self._report_stage(send, stage_name, announced)

            state = StoryState(
                theme=build_theme(description, request.moral, request.age),
                age=request.age,
                moral=request.moral,
                num_pages=self.num_pages,
                toy_description=build_toy_description(description),
                output_dir=run_settings.story_output_dir,
                started_at=datetime.now(timezone.utc),
            )

            pipeline = self._make_pipeline(
                settings=run_settings,
                reference_photo=request.photo,
                on_stage=on_stage,
            )
            state = await pipeline.run(state)
        except Exception as exc:  # noqa: BLE001 - обрыв фоновой задачи
            logger.error("Прогон сказки сорвался: {}", exc)
            await self._record(
                request, status=GenerationStatus.FAILED, error=str(exc)
            )
            await send(GENERATION_FAILED, [build_restart_keyboard().as_markup()])
            return
        finally:
            await self._close_quietly(pipeline)

        if state.status == "ok":
            await self._record(
                request,
                status=GenerationStatus.SUCCESS,
                tale_text=join_pages_text(state.pages),
            )
            await self._deliver_pages(send, state)
            await send(STORY_READY)
        else:
            await self._deliver_partial(request, send, state)

        logger.info(
            "Прогон {} завершён: страниц={} статус={} каталог={}",
            request.tale_id,
            len(state.pages),
            state.status,
            state.output_dir,
        )

    async def _deliver_pages(self, send: SendFunc, state: StoryState) -> None:
        """Отдать страницы по одной: «Страница N», текст и одна картинка.

        Одно вложение на сообщение — так текст страницы и её картинка всегда
        остаются рядом, а лимит мессенджера на несколько картинок в одном
        сообщении не мешает. Порядок — по номеру страницы, а не по порядку
        записи в состояние: нумерация книги должна совпадать с порядком
        чтения.
        """

        for page in sorted(state.pages, key=lambda item: item.page_num):
            await self._send_page(send, state, page.page_num, page.text)

    async def _deliver_partial(
        self,
        request: RunRequest,
        send: SendFunc,
        state: StoryState,
    ) -> None:
        """Отдать частичный результат и один раз предложить начать заново.

        Квота за такой прогон не начисляется: родитель получил не всю книжку,
        и деньги за неё уже потрачены. Повтор предлагается ровно один раз —
        иначе неограниченные переделки разошлись бы по цене.
        """

        missing = sorted(
            set(range(1, state.num_pages + 1))
            - {art.page_num for art in state.images}
        )
        reason = self._technical_reason(state)
        logger.warning(
            "Прогон {} частичный: страниц без картинки={} причина={}",
            request.tale_id,
            numbers_text(missing),
            reason,
        )
        await self._record(
            request,
            status=GenerationStatus.FAILED,
            tale_text=join_pages_text(state.pages),
            error=reason,
        )
        await self._deliver_pages(send, state)
        await send(
            PARTIAL_RESULT.format(numbers=numbers_text(missing)),
            [build_restart_keyboard().as_markup()],
        )
        if request.user_id in self._redo_offered:
            logger.info(
                "Повтор для пользователя {} уже предлагался, второй раз "
                "не предлагаем",
                request.user_id,
            )
            return
        self._redo_offered.add(request.user_id)
        await send(
            PARTIAL_REDO,
            [build_restart_keyboard().as_markup()],
        )

    @staticmethod
    def _technical_reason(state: StoryState) -> str:
        """Техническая причина статуса «требует проверки» из метрик прогона.

        Причина живёт в метриках этапов: пайплайн не поднимает исключение
        при частичном результате, поэтому текст ошибки нужно взять из
        последней метрики с ошибкой. Значение уходит только в лог и в запись
        о прогоне — в чате пользователь видит `PARTIAL_REASON`.
        """

        for metric in reversed(state.metrics):
            if metric.error:
                return metric.error
        return "прогон завершён с частичным результатом без метрики с ошибкой"

    async def _send_page(
        self,
        send: SendFunc,
        state: StoryState,
        page_num: int,
        text: str,
    ) -> None:
        """Отправить одну страницу: текст и, если есть, её картинку."""

        title = f"{PAGE_TITLE} {page_num}\n{text}"
        art = next(
            (item for item in state.images if item.page_num == page_num), None
        )
        if art is None:
            await send(f"{title}\n\n{PAGE_IMAGE_SKIPPED}")
            return

        try:
            attachment = _read_page_attachment(art.path)
        except OSError as exc:
            logger.warning("Файл страницы {} недоступен: {}", page_num, exc)
            await send(f"{title}\n\n{PAGE_WITHOUT_IMAGE}")
            return

        try:
            await send(title, [attachment])
        except Exception as exc:  # noqa: BLE001 - загрузка файла в MAX
            logger.warning(
                "Не удалось отправить страницу {}: {}", page_num, exc
            )
            # Отправка упала целиком, текст до пользователя не дошёл, поэтому
            # повторяем без вложения — иначе родитель потерял бы страницу.
            await send(f"{title}\n\n{PAGE_WITHOUT_IMAGE}")

    # --- Внутренняя реализация -------------------------------------------

    def _sender_for(self, chat_id: int) -> SendFunc:
        """Отправитель, зафиксированный на чат конкретного прогона.

        Замыкание хранит `chat_id` явно: после возврата обработчика событие
        MAX уже нельзя использовать для отправки.
        """

        async def send(
            text: str | None = None,
            attachments: list[Any] | None = None,
        ) -> Any:
            return await self._sender.send(
                chat_id, text=text, attachments=attachments
            )

        return send

    async def _report_stage(
        self,
        send: SendFunc,
        stage_name: str,
        announced: set[str],
    ) -> None:
        """Сообщить о прогрессе на границе этапа, не повторяя одно и то же.

        Повтор отсекается по тексту сообщения, а не по имени этапа: пять
        последовательных текстовых этапов иначе прислали бы пять одинаковых
        «Готовлю сказку…» подряд.
        """

        message = (
            ILLUSTRATION_PROGRESS
            if stage_name == ILLUSTRATE_STAGE
            else STORY_PROGRESS
        )
        if message in announced:
            return
        announced.add(message)
        await send(message)

    async def _record(
        self,
        request: RunRequest,
        *,
        status: GenerationStatus,
        tale_text: str | None = None,
        error: str | None = None,
    ) -> None:
        """Записать результат прогона в БД (статус, текст, ошибка)."""

        await self._repository.update_result(
            request.tale_id,
            status=status,
            tale_text=tale_text,
            error=error,
        )

    async def _notify_interrupted(self, chat_id: int) -> None:
        """Сообщить пользователю, что генерация прервана остановкой бота."""

        try:
            await self._sender.send(
                chat_id,
                text=RUN_INTERRUPTED,
                attachments=[build_restart_keyboard().as_markup()],
            )
        except Exception as exc:  # noqa: BLE001 - чат мог пропасть
            logger.warning("Не удалось сообщить о прерывании прогона: {}", exc)

    async def _run_guarded(self, request: RunRequest) -> None:
        """Обёртка фоновой задачи: сбой не должен остаться незамеченным."""

        try:
            await self.run(request)
        except asyncio.CancelledError:
            logger.info("Прогон {} отменён (остановка бота)", request.tale_id)
            raise
        except Exception as exc:  # noqa: BLE001 - задача без наблюдателя
            logger.error(
                "Непредвиденный сбой фонового прогона {}: {}",
                request.tale_id,
                exc,
            )
            try:
                await self._record(
                    request, status=GenerationStatus.FAILED, error=str(exc)
                )
            except Exception as nested:  # noqa: BLE001 - БД тоже может лежать
                logger.error("Не удалось записать сбой прогона: {}", nested)

    def _finish(self, user_id: int, task: asyncio.Task[None]) -> None:
        """Убрать завершённый прогон из реестра."""

        run = self._runs.get(user_id)
        if run is not None and run.task is task:
            del self._runs[user_id]

    @staticmethod
    async def _close_quietly(pipeline: StoryPipeline | None) -> None:
        """Закрыть провайдеры пайплайна, не дав сбою закрытия затереть статус.

        `None` — пайплайн не успел собраться, закрывать нечего.
        """

        if pipeline is None:
            return
        close = getattr(pipeline, "aclose", None)
        if not callable(close):
            return
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 - результат важнее чистоты
            logger.warning("Не удалось закрыть провайдеры прогона: {}", exc)


__all__ = [
    "ILLUSTRATE_STAGE",
    "PipelineFactory",
    "RunRequest",
    "StoryRunner",
    "build_story_pipeline",
]
