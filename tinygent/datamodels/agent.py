from abc import ABC
from abc import abstractmethod


class AbstractAgent(ABC):
    @abstractmethod
    def run(self, input_text: str) -> str:
        raise NotImplementedError('Subclasses must implement this method.')

    @property
    @abstractmethod
    def final_answer(self) -> str | None: ...
