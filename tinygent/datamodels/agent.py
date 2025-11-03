from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from tinygent.datamodels.agent_hooks import AgentHooks
from tinygent.types.builder import TinyModelBuildable

AgentType = TypeVar('AgentType', bound='AbstractAgent')


class AbstractAgentConfig(TinyModelBuildable[AgentType], Generic[AgentType]):
    """Abstract base class for agent configurations."""

    _agent_class: ClassVar

    def build(self) -> AgentType:
        """Build the agent instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class AbstractAgent(AgentHooks, ABC):
    """Abstract base class for agents."""

    @abstractmethod
    def run(self, input_text: str) -> str:
        """Run the agent with the given input text."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def run_stream(self, input_text: str) -> AsyncGenerator[str, None]:
        """Run the agent in streaming mode with the given input text."""
        raise NotImplementedError('Subclasses must implement this method.')
