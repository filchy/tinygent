from abc import ABC
from abc import abstractmethod
from typing import Any

from tiny_graph.driver.base import BaseDriver
from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.messages import BaseMessage


class BaseGraph(ABC):
    def __init__(
        self,
        llm: AbstractLLM,
        embedder: AbstractEmbedder,
        driver: BaseDriver,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.driver = driver

    @abstractmethod
    async def add_record(
        self,
        name: str,
        data: str | dict | BaseMessage,
        description: str,
        *,
        uuid: str | None = None,
        subgraph_id: str | None = None,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass
