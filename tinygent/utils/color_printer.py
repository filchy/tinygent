from typing import ClassVar


class TinyColorPrinter:
    COLORS: ClassVar[dict[str, str]] = {
        'RESET': '\033[0m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'YELLOW': '\033[93m',
        'GREEN': '\033[92m',
        'MAGENTA': '\033[95m',
        'RED': '\033[91m',
        'RED_BG': '\033[41m',
    }

    @classmethod
    def fmt(cls, label: str, color: str, msg: str) -> str:
        c = cls.COLORS.get(color.upper(), cls.COLORS['RESET'])
        return f'{c}[HOOK {label}]{cls.COLORS["RESET"]} {msg}'

    @classmethod
    def debug(cls, msg: str) -> str:
        return cls.fmt('DEBUG', 'CYAN', msg)

    @classmethod
    def info(cls, msg: str) -> str:
        return cls.fmt('INFO', 'GREEN', msg)

    @classmethod
    def warning(cls, msg: str) -> str:
        return cls.fmt('WARNING', 'YELLOW', msg)

    @classmethod
    def error(cls, msg: str) -> str:
        return cls.fmt('ERROR', 'RED', msg)

    @classmethod
    def custom(cls, label: str, msg: str, color: str = 'MAGENTA') -> str:
        return cls.fmt(label, color, msg)
