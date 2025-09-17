import logging

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


def setup_logger(log_level: str = 'info') -> logging.Logger:
    """Set up the logger for the application."""
    num_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d | %(name)-35s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(num_level)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    return root_logger
