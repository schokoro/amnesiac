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
фиксировать точную версию, например `amnesiac[select]==0.1.0`.

Публичный контракт и подробное описание поведения находятся в [docs/api.md](docs/api.md),
история изменений — в [CHANGELOG.md](CHANGELOG.md).
