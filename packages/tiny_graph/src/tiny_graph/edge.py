from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Any

from pydantic import Field

from tiny_graph.driver.base import BaseDriver
from tiny_graph.helper import generate_uuid
from tiny_graph.helper import get_current_timestamp
from tinygent.core.types.base import TinyModel


class TinyEdge(TinyModel, ABC):
    uuid: str = Field(
        description='unique edge identifier', default_factory=generate_uuid
    )

    name: str = Field(description='name of the edge, relation name')

    subgraph_id: str = Field(..., description='subgraph identifier')

    source_node_uuid: str

    target_node_uuid: str

    created_at: datetime = Field(default_factory=get_current_timestamp)

    attributes: dict[str, Any] = Field(
        default_factory=dict, description='Additional attributes of the node.'
    )

    @classmethod
    @abstractmethod
    def from_record(cls, record: dict) -> Any:
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    async def save(self, driver: BaseDriver) -> Any:
        raise NotImplementedError('Subclasses must implement this method.')
