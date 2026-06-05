import logging
from typing import Any
from typing import Generic
from typing import TypeVar

from pydantic import Field

from tinygent.core.datamodels.checkpointer import AbstractCheckpointer
from tinygent.core.datamodels.checkpointer import AbstractCheckpointerConfig

T = TypeVar('T', bound='TinyBaseCheckpointer')


logger = logging.getLogger(__name__)


class TinyBaseCheckpointerConfig(AbstractCheckpointerConfig[T], Generic[T]):
    type: Any = Field(default='base')

    data: dict[str, Any] = Field(default={})

    def build(self) -> T:
        """Build the BaseCheckpointer instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class TinyBaseCheckpointer(AbstractCheckpointer):
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def set_data(self, data: dict[str, Any]) -> None:
        self.data = data

    def setdefault(self, key: str, value: Any) -> Any:
        if key not in self.data:
            self.data[key] = value
        return self.data[key]

    def clear(self) -> None:
        self.data = {}

    def __getitem__(self, key: str, default: Any = None) -> Any:
        val = self.data.get(key, default)
        if val is None:
            logger.warning(
                'Key: %s is missing in checkpoint data %s', key, self.__class__.__name__
            )

        if default is not None:
            self.__setitem__('key', default)

        return val

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
