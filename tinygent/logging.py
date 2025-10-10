import logging

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[41m",  # Red background
    "RESET": "\033[0m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = COLORS.get(levelname, COLORS['RESET'])
        record.levelname = f'{color}{levelname}{COLORS['RESET']}'
        record.msg = f'{color}{record.msg}{COLORS['RESET']}'
        return super().format(record)


def setup_logger(log_level: str = "info") -> logging.Logger:
    """Set up the logger for the application with colors by level."""
    num_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)

    formatter = ColorFormatter(
        fmt="%(asctime)s.%(msecs)03d | %(name)-35s | %(levelname)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(num_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    return root_logger


def setup_general_loggers(log_level: str = 'warning') -> None:
    num_level = LOG_LEVELS.get(log_level.upper(), logging.WARNING)

    for name in ('httpx', 'httpcore', 'openai._base_client', 'asyncio'):
        logger = logging.getLogger(name)
        logger.setLevel(num_level)
