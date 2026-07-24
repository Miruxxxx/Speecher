"""Ask the local LM Studio model to find EVERY question in a transcript and
answer each one - the test for whether question detection can move off the
unreliable '?' the ASR model emits (see docs/NEMOTRON_PUNCTUATION_TODO.md).

Start LM Studio (a model loaded, server on) first, then:

    .venv\\Scripts\\python scripts\\llm_find_questions.py --file transcript.txt

With no --file it uses the built-in sample (the 64 s probe dialog). Streams the
model's answer so you can see whether it catches both questions - including the
"Почему тогда ... фетиш?" the model never punctuated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The Windows console defaults to cp866/cp1251 and turns Russian output into
# mojibake (and redirecting to a file bakes it in). Force UTF-8 on our own
# streams so the answer is readable either way.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# The app's real STREAM output for the 64 s dialog - punctuation as nemotron
# emitted it (note: only ONE '?', though there are two spoken questions).
SAMPLE = (
    "А вот как ты думаешь, у многих людей вообще есть комплекс насчет их палецев "
    "на ногах? Я имею в виду, что вот, например, какие-то то, что они носят "
    "какую-то область или на каблуках ходят, у них же дефрамируется какой то там "
    "положение бальчиков и ногтевая пластина, мне кажется, это выглядит страшно. "
    "Ты так не кажется, потому что мне не хочется иногда показывать свои ноги, но "
    "при этом футфетишь многих присутствует. Почему тогда футфетишь это настолько "
    "популярный фетишь, хороший вопрос, на который я не могу дать ответы, а еще не "
    "могут эти ответы на то, откуда у меня берутся синяки на ногах. Вот мне сейчас "
    "какую-то синяк на гулень в форме треугольник. Где я могла удариться формы "
    "треугольника себе вокруг. Объясните, пожалуйста"
)

PROMPT = (
    "Ниже — расшифровка устной речи (возможны ошибки распознавания и "
    "пропущенные знаки препинания, в том числе знаки вопроса). "
    "Найди ВСЕ вопросы, которые задаёт говорящий — по смыслу, а не по знаку «?». "
    "Для каждого выпиши сам вопрос и дай короткий, полезный ответ. "
    "Отвечай на русском.\n\nРасшифровка:\n{transcript}"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", type=Path, help="transcript .txt (UTF-8); default: built-in sample")
    args = ap.parse_args()

    transcript = args.file.read_text(encoding="utf-8") if args.file else SAMPLE

    from app_config import load_config
    from llm.lmstudio_client import LMStudioClient

    cfg = load_config(ROOT / "config" / "config.toml")
    client = LMStudioClient(cfg.llm)
    print(f"[llm] base_url: {cfg.llm.base_url}")
    if not client.is_available():
        print("[llm] LM Studio НЕ доступен — запусти сервер и загрузи модель, потом повтори.")
        return
    print(f"[llm] model: {client.resolve_model()}")
    print("[llm] --- ответ модели (стрим) ---\n")

    for token in client.stream_completion(
        [{"role": "user", "content": PROMPT.format(transcript=transcript)}]
    ):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()


if __name__ == "__main__":
    main()
