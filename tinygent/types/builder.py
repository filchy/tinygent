from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import TypeVar

from tinygent.types.base import TinyModel

T = TypeVar('T')


class TinyModelBuildable(TinyModel, Generic[T], ABC):
    @abstractmethod
    def build(self) -> Any:
        pass
