from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import AsyncGenerator
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from tinygent.agents.middleware.agent import TinyMiddlewareAgent
from tinygent.core.datamodels.memory import AbstractMemory
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.types.builder import TinyModelBuildable

AgentType = TypeVar('AgentType', bound='AbstractAgent')


class AbstractAgentConfig(TinyModelBuildable[AgentType], Generic[AgentType]):
    """Abstract base class for agent configurations."""

    _agent_class: ClassVar

    def build(self) -> AgentType:
        """Build the agent instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class AbstractAgent(TinyMiddlewareAgent, ABC):
    """Abstract base class for agents."""

    @property
    @abstractmethod
    def memory(self) -> AbstractMemory:
        """Get agents memory instance."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's internal state."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def run(
        self,
        input_text: str,
        *,
        run_id: str | None = None,
        reset: bool = True,
        history: list[AllTinyMessages] | None = None,
    ) -> str:
        """Run the agent with the given input text."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def run_stream(
        self,
        input_text: str,
        *,
        run_id: str | None = None,
        reset: bool = True,
        history: list[AllTinyMessages] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Run the agent in streaming mode with the given input text."""
        raise NotImplementedError('Subclasses must implement this method.')
