"""PawPal+ logging: one place that decides where log records go.

Every module logs through a child of the "pawpal" logger (pawpal.agent,
pawpal.qa, pawpal.guardrails, pawpal.model, pawpal.app), and only the app calls
setup_logging(). Library code never configures logging itself, so the tests and
any other caller decide for themselves what to do with the records.

Where records go once setup_logging() has run:
    logs/pawpal.log   - INFO and above (DEBUG with PAWPAL_LOG_LEVEL=DEBUG),
                        rotated at 1 MB with three old files kept
    stderr            - WARNING and above, so problems show in the terminal
                        running `streamlit run app.py`

What is logged, and what is not:
    - every agent run (with a short run id tying its lines together), each
      model call's latency, each tool call and its outcome, the post-check
    - every care-guide question: retrieval scores, the gate decision, status
    - every guardrail that fires (blocked input, flagged output, rate limit)
    - user text only as a short one-line preview (see preview()); API keys
      never, since nothing here ever sees one
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER = "pawpal"
DEFAULT_LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = "pawpal.log"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-16s %(message)s"
PREVIEW_CHARS = 80

_HANDLER_TAG = "_pawpal_handler"


def get_logger(name: str) -> logging.Logger:
    """The logger for one part of PawPal+, e.g. get_logger("agent") -> pawpal.agent."""
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def setup_logging(log_dir: str | Path | None = None, level: str | int | None = None) -> Path:
    """Send PawPal+ log records to a rotating file and warnings to stderr.

    Safe to call on every Streamlit rerun: handlers are only added once per
    process. The level comes from the argument, else PAWPAL_LOG_LEVEL, else
    INFO.

    Returns:
        The path of the log file.
    """
    path = Path(log_dir or os.environ.get("PAWPAL_LOG_DIR") or DEFAULT_LOG_DIR) / LOG_FILE
    logger = logging.getLogger(ROOT_LOGGER)
    level = level or os.environ.get("PAWPAL_LOG_LEVEL", "INFO")
    logger.setLevel(level.upper() if isinstance(level, str) else level)

    if any(getattr(handler, _HANDLER_TAG, False) for handler in logger.handlers):
        return path

    formatter = logging.Formatter(LOG_FORMAT)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    except OSError as exc:  # read-only folder: keep the app running, log to stderr only
        file_handler = None
        logging.getLogger(ROOT_LOGGER).warning("Could not open log file %s: %s", path, exc)
    if file_handler is not None:
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_TAG, True)
        logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    setattr(console, _HANDLER_TAG, True)
    logger.addHandler(console)
    return path


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """User text squeezed onto one short line, for log messages.

    Enough to recognise a request in the log without copying whole messages
    into it.
    """
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
