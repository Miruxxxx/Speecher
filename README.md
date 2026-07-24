# Speecher

Десктопное приложение (Windows) для **транскрипции системного звука в реальном времени**: захватывает всё, что играет через колонки/наушники (WASAPI loopback), прогоняет через faster-whisper со стабилизацией LocalAgreement-2 и показывает текст в полупрозрачном always-on-top оверлее. По кнопке — саммари разговора и ответы на последний прозвучавший вопрос через локальную LLM (LM Studio).

Типичный сценарий: онлайн-созвон/лекция → живой транскрипт поверх окон → «Sum 5min» даёт выжимку, «Last ? + AI» отвечает на вопрос, который вы прослушали.

## Стек

| Слой | Технология |
|---|---|
| Захват звука | нативный бинарь на Rust (WASAPI loopback напрямую) **в отдельном процессе** с супервизором и авто-рестартом; downmix и ресемплинг → 16 kHz там же |
| ASR | два переключаемых движка (`[asr] engine`): **whisper** (по умолчанию) — faster-whisper `large-v3-turbo` на CUDA fp16 + LocalAgreement-2 с гейтом тишины и фильтрами галлюцинаций; **nemotron** — `nvidia/nemotron-3.5-asr-streaming-0.6b`, cache-aware стриминг, ~0.7 с до слова, пунктуация нативная |
| Пунктуация | silero-te (ru/en/de/es, CPU) — только для движка whisper; nemotron пунктуирует сам |
| LLM | LM Studio через OpenAI-совместимый REST (`localhost:1234`) |
| UI | PyQt6 — прозрачный frameless-оверлей + консольный вывод |

Python 3.11, только Windows (WASAPI loopback).

## Установка

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
cargo build --release --manifest-path native\audio_capture\Cargo.toml
```

torch ставится первым и отдельным индексом: с PyPI приедет `+cpu`, и движок
nemotron работать не будет (подробности — в шапке `requirements.txt`).

Последняя строка собирает захват звука — без неё приложение стартует, но сразу
скажет, что бинарь не найден. Нужен [rustup](https://rustup.rs) с
`stable-x86_64-pc-windows-msvc`; линкер берётся из Visual Studio Build Tools.

Требования:
- NVIDIA GPU + драйвер для faster-whisper на CUDA (либо `device = "cpu"` + `compute_type = "int8"` в конфиге — медленнее).
- Для LLM-фич: запущенный [LM Studio](https://lmstudio.ai) с загруженной моделью и сервером на :1234. Без него всё работает, LLM-кнопки покажут предупреждение.
- Первый запуск скачивает модели: whisper `large-v3-turbo` (~1.6 GB с HuggingFace) и silero-te (~50 MB). Окно появляется сразу, прогресс виден в статусной строке оверлея.

## Запуск

```powershell
python -m src
```

## Конфигурация

Все настройки — в [config/config.toml](config/config.toml) (файл можно удалить — приложение запустится на дефолтах). Ключевое:

- `[audio] device_hint` — подстрока имени устройства вывода, чей loopback слушаем (`"HyperX"`); пусто = устройство по умолчанию. `source = "mic"` переключает захват на микрофон, `binary` — путь к бинарю захвата, если он лежит не в дереве проекта.
- `[asr] engine` — `whisper` (по умолчанию) или `nemotron`; неизвестное значение → `whisper` с предупреждением. Ключи ниже относятся к whisper-пути.
- `[asr.nemotron]` — читается только при `engine = "nemotron"`: модель, `dtype` (`float32` по умолчанию — fp16 экономит VRAM, но декодирует вдвое медленнее), `lookahead_tokens` `0/3/6/13` → задержка 80/320/560/1120 мс.
- `[asr] model / device / language` — модель Whisper, cuda/cpu и режим языка:
  - `"auto"` — определить по первой уверенной речи и зафиксировать (по умолчанию);
  - `"ru"`, `"en"`, … — жёстко;
  - `""` — переопределять каждый цикл (не рекомендуется: даёт «кашу» алфавитов).
- `[asr.filters]` — пороги отбраковки галлюцинаций (`no_speech_prob`, `avg_logprob`), схлопывание повторов, гейт тишины.
- `[punctuation] backend` — `silero` (ru/en/de/es, CPU) / `off`. При `engine = "nemotron"` воркер не запускается вообще.
- `[llm]` — URL LM Studio, таймауты; `model = ""` берёт первую загруженную в LM Studio модель.

## Управление оверлеем

- Перетаскивание — зажать ЛКМ в любом месте окна; ресайз — грип в правом нижнем углу; закрыть — ✕ или Esc (закрытие останавливает всё приложение).
- **Last ?** — найти последний вопрос в транскрипте (по знаку «?»).
- **Last ? + AI** — то же + ответ LLM с контекстом.
- **Sum 5min / Sum Xmin…** — саммари за последние N минут.

## Структура проекта

```
config/
  config.toml              # все настройки (TOML)
native/
  audio_capture/           # захват звука на Rust (WASAPI): бинарь + его тесты
src/
  main.py                  # сборка пайплайна: окно сразу, модели в фоне
  app_config.py            # загрузка config.toml поверх дефолтов
  audio/
    rust_capture.py        # спавн/рестарт нативного ребёнка, байты → буфер
    capture_base.py        # политика рестарта, запись в буфер/frame_sink
    buffer.py              # GrowingAudioBuffer: append-only + head-trim
  asr/
    backends.py            # протокол ASRBackend + выбор движка по [asr] engine
    whisper_backend.py     # pull-путь: снапшоты буфера → StreamingASREngine
    streaming_engine.py    # LocalAgreement-2 + гейт тишины (поток engine)
    whisper_adapter.py     # faster-whisper → Word[]; sticky-язык, фильтры галлюцинаций
    nemotron_backend.py    # push-путь: фреймы из frame_sink → стриминговый RNN-T
    punctuation_backends.py# silero-te (CPU) | off
    punctuation_worker.py  # фоновая пере-пунктуация хвоста транскрипта (только whisper)
    events.py              # ASREvent (commit/partial/log/fatal), Word
  store/
    transcript_store.py    # потокобезопасный стор слов + запросы (вопрос/период)
  llm/
    engine.py              # очередь LLM-задач в отдельном потоке
    lmstudio_client.py     # LM Studio REST: без ретраев, короткий health-check
  ui/
    overlay.py             # оверлей: транскрипт, статус, LLM-панель, кнопки
    output_sink.py         # консольный рендер (ANSI), параллельно оверлею
  utils/                   # normalize_word, LatencyTracker
tests/                     # pytest: buffer, engine, store, adapter, worker, config
scripts/                   # ручные утилиты (список устройств, уровни звука)
docs/                      # архитектура, код-ревью, роадмап
storage/transcripts/       # (зарезервировано под персист транскриптов)
```

Подробности потоков и алгоритма — в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Известные проблемы

1. Захват требует собранного бинаря (`cargo build --release …`): без него приложение сразу скажет, чего не хватает, — запасного пути на питоновских библиотеках нет. Раньше эту роль играл PyAudioWPatch, который портил кучу своего процесса после нескольких минут простаивающего loopback-стрима (история — [docs/CODE_REVIEW_2026-07-15.md](docs/CODE_REVIEW_2026-07-15.md) §B1); он удалён вместе с этим багом.
2. WASAPI loopback отдаёт фреймы только пока что-то играет: в полной тишине захват «молчит» — это нормально.
3. Модель `large-v3-turbo` на слабом GPU может не укладываться в 1-секундный цикл — поставьте `small`/`base` в конфиге.

## Диагностика

- Список устройств: `python scripts/list_audio_devices.py`
- Долгий прогон захвата: `python scripts/capture_soak.py --seconds 900`
- `[ui] console_verbose_logs = true` — показывает log-события движка (рестарты захвата, force-commit) в консоли.
- Тесты: `python -m pytest tests/ -q`
- Крэши смотреть в Windows Event Log: `Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='Application Error'} | ? Message -match 'python'`
