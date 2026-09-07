# amnesiac

`amnesiac` — библиотека переиспользуемой LLM-логики, извлечённой из исследовательского
пайплайна: двухстадийная суммаризация новостей и чистая векторная математика для отбора
документов. Библиотека никогда не обращается к базе данных, сервису эмбеддингов, файлам
конфигурации, файловой системе или переменным окружения. Она получает данные аргументами и
возвращает данные; весь ввод-вывод, кроме вызовов LLM API, остаётся на стороне потребителя.

## Установка

Для суммаризации (дополнительный extra не нужен):

```shell
pip install amnesiac
```

Для отбора документов через `amnesiac.select` установите extra `amnesiac[select]`, который
добавляет NumPy:

```shell
pip install 'amnesiac[select]'
```

## Суммаризация

Клиент `AsyncOpenAI`, модель, данные и ограничитель параллелизма предоставляет потребитель.
Пакет работает в принадлежащем вызывающему коду цикле событий и не создаёт его сам.

```python
from openai import AsyncOpenAI

from amnesiac import Doc
from amnesiac.summarize import PromptPack, SummarizeConfig, summarize
from amnesiac.summarize.prompts import RU_MACRO_V1

client = AsyncOpenAI(api_key="example-api-key")
docs = [
    Doc(
        text="Банк сохранил ключевую ставку без изменений.",
        channel="example-news",
        day_number=3,
        doc_id="news-42",
    )
]
prompts: PromptPack = RU_MACRO_V1.bind(horizon_days=14)
config = SummarizeConfig(temperature=0.3, max_failed_axes=0)


async def make_summary():
    return await summarize(
        client=client,
        model="provider/model-name",
        axes={"денежно-кредитная политика": docs},
        prompts=prompts,
        config=config,
    )
```

`make_summary()` должен вызывать код потребителя внутри уже работающего цикла событий. Его
результат содержит итоговое мета-саммари, саммари по осям, сведения об ошибках осей и usage.

Если часть осей уже лежит в кеше потребителя, можно досчитать только недостающие и отдельно
собрать мета-саммари:

```python
from amnesiac.summarize import summarize_axes, summarize_meta


async def make_summary_from_cache(client, model, axes, cached, prompts, limiter, config):
    missing = {name: docs for name, docs in axes.items() if name not in cached}

    if missing:
        fresh = await summarize_axes(
            client=client,
            model=model,
            axes=missing,
            prompts=prompts,
            limiter=limiter,
            config=config,
        )
        ready = cached | fresh.axis_summaries
    else:
        ready = cached

    merged = {name: ready[name] for name in axes}

    return await summarize_meta(
        client=client,
        model=model,
        axis_summaries=merged,
        prompts=prompts,
        limiter=limiter,
        config=config,
    )
```

При вызове `summarize()` исключение `MetaSummaryError` поднимается, только если мета-вызов
исчерпал попытки на пустом ответе, и несёт результат первой стадии целиком:
`axis_summaries`, `failed_axes`, `axis_errors` и `usage`. Провайдерская ошибка после
исчерпания ретраев проходит наружу без изменений и не несёт частичного результата.
У пойманного `MetaSummaryError` в `err.axis_summaries` уже лежат саммари в правильном порядке
ключей. Передача их в `summarize_meta()` делает повтор одним вызовом провайдера вместо
повторного расчёта всех осей.
Чтобы сохранить первую стадию и на этом пути, вызывайте стадии раздельно: сначала
`summarize_axes()`, чтобы саммари были уже на руках до возможного сбоя мета-вызова,
затем `summarize_meta()`.

Кешировать можно только оси, отсутствующие в `failed_axes`: для упавшей оси
`axis_summaries` содержит `failed_axis_placeholder`, и сохранение этой заглушки как результата
навсегда оставит дату с дыркой, которая выглядит как успешный расчёт.

## Отбор документов

Этот пример требует установки `amnesiac[select]` и выполняется целиком без сетевых вызовов:

```python
import numpy as np

from amnesiac.select import select_by_axis

vectors = np.array(
    [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ]
)
queries = {"инфляция": np.array([1.0, 0.0])}

selected = select_by_axis(
    vectors,
    queries,
    texts=["кратко", "более длинный дубликат", "другая тема"],
    top_k=3,
    dedup_threshold=0.95,
)
print(selected)
# {'инфляция': [0, 2]}
```

Документы, сходство которых достигает `dedup_threshold`, дедуплицируются с сохранением более
короткого текста — поэтому индекс 1 (`"более длинный дубликат"`) отсутствует в результате, а
индекс 0 (`"кратко"`) остаётся.

Функция не приводит dtype и считает в типе, переданном потребителем; при близких значениях
сходства порядок зависит от dtype, поэтому для воспроизведения существующего отбора нужно
передавать `float32` и в `X`, и в `queries` (см. [§6 публичного
контракта](docs/api.md#6-amnesiacselect)).

`amnesiac.select` нестабилен на всём протяжении `0.x`; потребителю этой подсистемы следует
фиксировать точную версию, например `amnesiac[select]==0.2.0`.

Публичный контракт и подробное описание поведения находятся в [docs/api.md](docs/api.md),
история изменений — в [CHANGELOG.md](CHANGELOG.md).
