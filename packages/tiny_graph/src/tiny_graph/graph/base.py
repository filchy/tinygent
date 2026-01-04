from abc import ABC
from abc import abstractmethod

from tiny_graph.driver.base import BaseDriver
from tinygent.datamodels.embedder import AbstractEmbedder
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.messages import BaseMessage


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
    ) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass
