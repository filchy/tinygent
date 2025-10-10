from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

from pydantic import ConfigDict, PrivateAttr

from tinygent.types.base import TinyModel

AgentType = TypeVar('AgentType', bound='AbstractAgent')


class AbstractAgentConfig(TinyModel, Generic[AgentType], ABC):
    """Abstract base class for agent configurations."""

    agent_type: str  # used as discriminator

    _agent_class: type[AgentType] = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def build_agent(self) -> AgentType:
        """Build and return an agent instance."""
        raise NotImplementedError('Subclasses must implement this method.')


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
