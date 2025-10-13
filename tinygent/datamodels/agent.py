from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from pydantic import ConfigDict

from tinygent.types.builder import TinyModelBuildable

AgentType = TypeVar('AgentType', bound='AbstractAgent')


class AbstractAgentConfig(TinyModelBuildable[AgentType], Generic[AgentType]):
    """Abstract base class for agent configurations."""

    type: Any  # used as discriminator

    _discriminator_field: ClassVar[str] = 'type'

    _agent_class: ClassVar

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_discriminator_field(cls) -> str:
        """Get the name of the discriminator field."""
        return cls._discriminator_field


class AbstractAgent(ABC):
    """Abstract base class for agents."""

    @abstractmethod
    def run(self, input_text: str) -> str:
        """Run the agent with the given input text."""
        raise NotImplementedError('Subclasses must implement this method.')

    @property
    @abstractmethod
    def final_answer(self) -> str | None:
        """Get the final answer produced by the agent, if any."""
        raise NotImplementedError('Subclasses must implement this method.')
