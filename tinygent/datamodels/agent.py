from abc import ABC
from abc import abstractmethod


class AbstractAgent(ABC):
    """Abstract base class for agents."""

    @abstractmethod
    def run(self, input_text: str) -> str:
        """Run the agent with the given input text."""
        raise NotImplementedError('Subclasses must implement this method.')

    @property
    @abstractmethod
    def final_answer(self) -> str | None: ...
    """Get the final answer produced by the agent, if any."""
