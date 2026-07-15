# Speecher

Десктопное приложение (Windows) для **транскрипции системного звука в реальном времени**: захватывает всё, что играет через колонки/наушники (WASAPI loopback), прогоняет через faster-whisper со стабилизацией LocalAgreement-2 и показывает текст в полупрозрачном always-on-top оверлее. По кнопке — саммари разговора и ответы на последний прозвучавший вопрос через локальную LLM (LM Studio).

Типичный сценарий: онлайн-созвон/лекция → живой транскрипт поверх окон → «Sum 5min» даёт выжимку, «Last ? + AI» отвечает на вопрос, который вы прослушали.

## Стек

| Слой | Технология |
|---|---|
| Захват звука | PyAudioWPatch (WASAPI loopback), scipy (ресемплинг → 16 kHz) |
| ASR | faster-whisper (`base`, CUDA fp16) + собственный стриминг-движок LocalAgreement-2 |
| Пунктуация | deepmultilingualpunctuation (xlm-roberta, опционально) |
| LLM | LM Studio через OpenAI-совместимый REST (`localhost:1234`) |
| UI | PyQt6 — прозрачный frameless-оверлей + консольный вывод |

Python 3.11, только Windows (WASAPI loopback).

## Установка

```powershell
pip install -r requirements.txt
# torch с CUDA (нужен только для пунктуации):
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Требования:
- NVIDIA GPU + драйвер (faster-whisper работает на CUDA; CPU-фоллбэка в коде сейчас нет).
- Для LLM-фич: запущенный [LM Studio](https://lmstudio.ai) с загруженной моделью и включённым сервером на порту 1234. Без него приложение работает, LLM-кнопки показывают предупреждение.
- Первый запуск с пунктуацией скачивает модель ~2.1 GB с HuggingFace — окно появится только после загрузки (см. Известные проблемы).

## Запуск

```powershell
python -m src
```

Настройки пока захардкожены в [src/main.py](src/main.py): `name_hint="HyperX"` (подстрока имени устройства вывода), модель `base`, `ADD_PUNCTUATION = True`.

## Управление оверлеем

- Перетаскивание — зажать ЛКМ в любом месте окна; ресайз — грип в правом нижнем углу; закрыть — ✕ или Esc (закрытие останавливает всё приложение).
- **Last ?** — найти последний вопрос в транскрипте (по знаку «?»).
- **Last ? + AI** — то же + ответ LLM с контекстом.
- **Sum 5min / Sum Xmin…** — саммари за последние N минут.

## Структура проекта

```
src/
  main.py                  # сборка пайплайна, Qt main loop
  __main__.py              # запуск через python -m src
  audio/
    capture_stream.py      # WASAPI loopback → GrowingAudioBuffer (поток capture)
    buffer.py              # GrowingAudioBuffer: append-only + head-trim по глобальному времени
    devices.py             # выбор loopback-устройства по подстроке имени
  asr/
    streaming_engine.py    # StreamingASREngine: LocalAgreement-2 (поток engine)
    whisper_adapter.py     # faster-whisper → list[Word] с word-level таймстемпами
    events.py              # ASREvent (commit/partial/log/fatal), Word
    punctuation_worker.py  # фоновая пере-пунктуация хвоста транскрипта
  store/
    transcript_store.py    # потокобезопасный стор закоммиченных слов + запросы
  llm/
    engine.py              # очередь LLM-задач в отдельном потоке
    lmstudio_client.py     # обёртка над LM Studio REST (стриминг токенов)
  ui/
    overlay.py             # PyQt6-оверлей: транскрипт, партиалы, LLM-панель, кнопки
    output_sink.py         # консольный рендер (ANSI), работает параллельно оверлею
  utils/
    text.py                # normalize_word для сравнения гипотез
    latency.py             # rolling-window метрики задержек

  transcript_engine.py     # [ЛЕГАСИ] старый движок до переписывания — не используется
  asr/streaming_worker.py  # [ЛЕГАСИ] то же

tests/                     # УСТАРЕЛИ: написаны под легаси transcript_engine
scripts/                   # ручные утилиты (список устройств, уровни звука)
docs/                      # архитектура, код-ревью, роадмап
```

Подробности потоков и алгоритма — в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Известные проблемы

1. **Нативный крэш 0xc0000005 через ~2–8 минут работы** — виновник локализован: PyAudioWPatch (PortAudio WASAPI loopback) рушит кучу процесса, когда loopback-стрим долго простаивает без системного звука. Воспроизводится изолированным захватом без Whisper/Qt; Whisper-цикл в одиночку чист (5438 декодов без сбоя). План лечения — в [docs/CODE_REVIEW_2026-07-15.md](docs/CODE_REVIEW_2026-07-15.md), §B1. Это главный блокер.
2. **Автоопределение языка скачет** — `language=None` заставляет Whisper угадывать язык на каждом ~1-секундном цикле: на шуме/акценте язык прыгает (en→de→ru), в транскрипт попадает «каша» не на том языке. Лечится фиксацией языка в `WhisperAdapter(language="ru"|"en")`.
3. **Долгий первый запуск** — вся инициализация (Whisper, HF-проверки, модель пунктуации) идёт до показа окна; на холодном кэше HuggingFace это минуты без какого-либо UI.
4. **UI подвисает без LM Studio** — проверка доступности сервера выполняется в UI-потоке с ретраями openai-клиента.
5. **Модель пунктуации не знает русского** (fullstop-punctuation-multilang-large: EN/DE/FR/IT) и занимает VRAM (грузится на GPU при наличии CUDA).
6. Тесты в `tests/` устарели (легаси-движок) и падают на импорте.

## Диагностика

- Список loopback-устройств: `python scripts/list_audio_devices.py`
- Логи: приложение пишет INFO-логи в stderr; консольный sink дублирует коммиты в stdout.
- Крэши смотреть в Windows Event Log: `Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='Application Error'} | ? Message -match 'python'`
