from __future__ import annotations

import os
import sys


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.join(root, "src")
    sys.path.insert(0, src)

    from main import main as app_main  # import after sys.path tweak

    app_main()


if __name__ == "__main__":
    main()

