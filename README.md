# Speecher

Десктопное приложение (Windows) для **транскрипции системного звука в реальном времени**: захватывает всё, что играет через колонки/наушники (WASAPI loopback), прогоняет через faster-whisper со стабилизацией LocalAgreement-2 и показывает текст в полупрозрачном always-on-top оверлее. По кнопке — саммари разговора и ответы на последний прозвучавший вопрос через локальную LLM (LM Studio).

Типичный сценарий: онлайн-созвон/лекция → живой транскрипт поверх окон → «Sum 5min» даёт выжимку, «Last ? + AI» отвечает на вопрос, который вы прослушали.

## Стек

| Слой | Технология |
|---|---|
| Захват звука | PyAudioWPatch (WASAPI loopback) **в отдельном подпроцессе** с супервизором и авто-рестартом; soxr (стриминговый ресемплинг → 16 kHz) |
| ASR | faster-whisper (`large-v3-turbo` по умолчанию, CUDA fp16) + стриминг-движок LocalAgreement-2 с гейтом тишины и фильтрами галлюцинаций |
| Пунктуация | silero-te (ru/en/de/es, CPU) — по умолчанию; deepmultilingualpunctuation — опция |
| LLM | LM Studio через OpenAI-совместимый REST (`localhost:1234`) |
| UI | PyQt6 — прозрачный frameless-оверлей + консольный вывод |

Python 3.11, только Windows (WASAPI loopback).

## Установка

```powershell
pip install -r requirements.txt
# torch с CUDA не обязателен: пунктуация работает на CPU.
```

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

- `[audio] device_hint` — подстрока имени устройства вывода, чей loopback слушаем (`"HyperX"`).
- `[asr] model / device / language` — модель Whisper, cuda/cpu и режим языка:
  - `"auto"` — определить по первой уверенной речи и зафиксировать (по умолчанию);
  - `"ru"`, `"en"`, … — жёстко;
  - `""` — переопределять каждый цикл (не рекомендуется: даёт «кашу» алфавитов).
- `[asr.filters]` — пороги отбраковки галлюцинаций (`no_speech_prob`, `avg_logprob`), схлопывание повторов, гейт тишины.
- `[punctuation] backend` — `silero` (ru/en/de/es, CPU) / `deepmultilingual` (en/de/fr/it) / `off`.
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
src/
  main.py                  # сборка пайплайна: окно сразу, модели в фоне
  app_config.py            # загрузка config.toml поверх дефолтов
  audio/
    capture_process.py     # ДОЧЕРНИЙ ПРОЦЕСС: PortAudio loopback → mp.Queue
    capture_supervisor.py  # спавн/рестарт ребёнка, mono+soxr → буфер
    buffer.py              # GrowingAudioBuffer: append-only + head-trim
    devices.py             # выбор loopback-устройства по подстроке имени
  asr/
    streaming_engine.py    # LocalAgreement-2 + гейт тишины (поток engine)
    whisper_adapter.py     # faster-whisper → Word[]; sticky-язык, фильтры галлюцинаций
    punctuation_backends.py# silero-te | fullstop (CPU) | off
    punctuation_worker.py  # фоновая пере-пунктуация хвоста транскрипта
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

1. **PyAudioWPatch (PortAudio) портит кучу своего процесса** после нескольких минут простаивающего loopback-стрима — поэтому захват вынесен в одноразовый дочерний процесс: его падение переживается авто-рестартом (пауза `restart_backoff_sec`, в тишине данные всё равно не идут). Корневая причина остаётся в библиотеке; альтернативы — в [docs/ROADMAP.md](docs/ROADMAP.md).
2. WASAPI loopback отдаёт фреймы только пока что-то играет: в полной тишине захват «молчит» — это нормально.
3. Модель `large-v3-turbo` на слабом GPU может не укладываться в 1-секундный цикл — поставьте `small`/`base` в конфиге.

## Диагностика

- Список loopback-устройств: `python scripts/list_audio_devices.py`
- `[ui] console_verbose_logs = true` — показывает log-события движка (рестарты захвата, force-commit) в консоли.
- Тесты: `python -m pytest tests/ -q`
- Крэши смотреть в Windows Event Log: `Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='Application Error'} | ? Message -match 'python'`
