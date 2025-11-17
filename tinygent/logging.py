import logging

from tinygent.utils.color_printer import TinyColorPrinter

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


class ColorFormatter(logging.Formatter):
    level_map = {
        'DEBUG': 'CYAN',
        'INFO': 'GREEN',
        'WARNING': 'YELLOW',
        'ERROR': 'RED',
        'CRITICAL': 'RED_BG',
    }

    def format(self, record: logging.LogRecord) -> str:
        color = TinyColorPrinter.COLORS.get(
            self.level_map.get(record.levelname, ''), TinyColorPrinter.COLORS['RESET']
        )
        reset = TinyColorPrinter.COLORS['RESET']

        record.levelname = f'{color}{record.levelname}{reset}'
        record.msg = f'{color}{record.msg}{reset}'

        return super().format(record)


def setup_logger(
    log_level: str = 'info', general_log_level: str = 'warning'
) -> logging.Logger:
    """Set up the logger for the application with colors by level."""
    num_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)

    formatter = ColorFormatter(
        fmt='%(asctime)s.%(msecs)03d | %(name)-35s | %(levelname)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(num_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    setup_general_loggers(general_log_level)

    return root_logger


def setup_general_loggers(log_level: str = 'warning') -> None:
    num_level = LOG_LEVELS.get(log_level.upper(), logging.WARNING)

    for name in ('httpx', 'httpcore', 'openai._base_client', 'asyncio'):
        logger = logging.getLogger(name)
        logger.setLevel(num_level)
