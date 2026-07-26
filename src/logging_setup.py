"""Logging that survives the session it describes.

The console handler is the one that has always been here. The file handler is
the point: this app runs for hours behind other windows, and the things that go
wrong in it are quiet — an ASR engine that fell back to whisper, a journal whose
disk went away, an API key that stopped working. All of those announce
themselves in the status zone, which is ~125 px wide and clears itself after a
few seconds. Started from a shortcut rather than a terminal, the app used to
leave no trace of any of it.

Rotation is deliberately small (2 MB × 3). This is a tail to read after
something looked wrong, not an archive to grep a month later.

Failing to open the log file is never fatal: the app logs the reason to the
console and runs without a file. A diagnostic aid that can stop the program is
worse than no diagnostic aid.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from app_config import LoggingConfig

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
FILENAME = "speecher.log"

logger = logging.getLogger(__name__)


def setup(
    cfg: Optional[LoggingConfig] = None,
    *,
    root: Optional[Path] = None,
    level: int = logging.INFO,
) -> Optional[Path]:
    """Install the console and (if configured) the rotating file handler.

    Returns the log file's path, or None when only the console is active.
    Safe to call twice: handlers this module installed are replaced, not
    stacked, so a second call does not double every line.
    """
    cfg = cfg or LoggingConfig()
    formatter = logging.Formatter(FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        if getattr(handler, "_speecher", False):
            root_logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console._speecher = True  # type: ignore[attr-defined]
    root_logger.addHandler(console)

    directory = (cfg.dir or "").strip()
    if not directory:
        return None

    path = Path(directory)
    if not path.is_absolute():
        path = (root or Path(__file__).resolve().parent.parent) / path
    try:
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path / FILENAME,
            maxBytes=max(0, int(cfg.max_bytes)),
            backupCount=max(0, int(cfg.backups)),
            encoding="utf-8",
        )
    except OSError as exc:
        # No log file is a degraded state, not a broken one.
        logger.warning("logging: cannot open %s (%s); console only", path, exc)
        return None

    file_handler.setFormatter(formatter)
    file_handler._speecher = True  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)
    return path / FILENAME


def banner(log_path: Optional[Path], config_path: Path) -> None:
    """One line that makes a log file self-describing.

    Whoever opens the file later needs to know which run and which config
    produced it before anything below it means much.
    """
    logging.getLogger("main").info(
        "Speecher starting: python=%s, config=%s, log=%s",
        sys.version.split()[0],
        config_path,
        log_path or "console only",
    )
