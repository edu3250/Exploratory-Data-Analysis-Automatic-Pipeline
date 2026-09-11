"""
Logging utilities with correlation IDs and structured logging.
"""

import json
import logging
import logging.handlers
import sys
import uuid
from datetime import datetime
from pathlib import Path


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for the run."""
    return str(uuid.uuid4())[:8]


def setup_logging(log_dir: Path, correlation_id: str, verbose: bool = False) -> tuple[logging.Logger, str]:
    """
    Set up logging with file and console handlers.

    Args:
        log_dir: Directory to store log files
        correlation_id: Unique ID for this run
        verbose: If True, log at DEBUG level

    Returns:
        (logger, log_file_path)
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eda_{timestamp}_{correlation_id}.log"

    logger = logging.getLogger("eda_pipeline")
    for handler in logger.handlers[:]:
        # Close explicitly before dropping the reference: otherwise the
        # underlying file descriptor (this logger gets a fresh FileHandler on
        # every pipeline run) stays open until garbage collection, which can
        # surface much later as an unrelated "Exception ignored in ..." at
        # interpreter/GC time (very visible under `pytest -W error`, and
        # generally just poor resource hygiene in long-lived processes).
        handler.close()
    logger.handlers.clear()
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(correlation_id)s] %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(correlation_id)s] %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Add a filter to inject correlation_id
    class CorrelationIdFilter(logging.Filter):
        def __init__(self, corr_id: str):
            self.corr_id = corr_id

        def filter(self, record: logging.LogRecord) -> bool:
            record.correlation_id = self.corr_id
            return True

    corr_filter = CorrelationIdFilter(correlation_id)
    console_handler.addFilter(corr_filter)
    file_handler.addFilter(corr_filter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, str(log_file)


class StructuredLogWriter:
    """Write structured logs (JSON) for machine parsing."""

    def __init__(self, log_file: Path, correlation_id: str):
        self.log_file = log_file
        self.correlation_id = correlation_id

    def write(self, event_type: str, data: dict) -> None:
        """Write a structured log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": self.correlation_id,
            "event_type": event_type,
            **data,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
