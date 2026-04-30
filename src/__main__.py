from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # Allow running the project with `python -m src` without forcing a packaging refactor.
    src_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(src_dir))

    from main import main as app_main  # import after sys.path tweak

    app_main()


if __name__ == "__main__":
    main()
