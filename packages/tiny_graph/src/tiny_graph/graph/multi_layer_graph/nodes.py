from __future__ import annotations

from datetime import datetime

from pydantic import TypeAdapter
from pydantic import model_validator
from typing_extensions import Self

from tiny_graph.driver.base import BaseDriver
from tiny_graph.graph.multi_layer_graph.types import DataType
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.graph.multi_layer_graph.utils.model_repr import compact_model_repr
from tiny_graph.helper import parse_db_date
from tiny_graph.node import TinyNode
from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.datamodels.messages import BaseMessage


class TinyEventNode(TinyNode):
    type = NodeType.EVENT

    data: str | dict | BaseMessage

    description: str

    data_type: DataType

    valid_at: datetime

    @model_validator(mode='after')
    def validate_(self) -> Self:
        type_map = {
            DataType.TEXT: str,
            DataType.JSON: dict,
            DataType.MESSAGE: BaseMessage,
        }

        if (desired_type := type_map.get(self.data_type)) is not (
            curr_type := type(self.data)
        ):
            raise TypeError(
                f'If node data is of type: {self.data_type} then data must be of type {desired_type} but is {curr_type}'
            )
        return self

    @property
    def serialized_data(self) -> str | dict:
        if isinstance(self.data, BaseMessage):
            return self.data.model_dump(mode='json')
        return self.data

    @staticmethod
    def _parse_data(data: str | dict, type_: DataType) -> str | dict | BaseMessage:
        match type_:
            case DataType.TEXT.value:
                if not isinstance(data, str):
                    raise TypeError(f'Expected str for TEXT, got {type(data)}')
                return data

            case DataType.JSON.value:
                if not isinstance(data, dict):
                    raise TypeError(f'Expected dict for JSON, got {type(data)}')
                return data

            case DataType.MESSAGE.value:
                if not isinstance(data, dict):
                    raise TypeError(f'Expected dict for MESSAGE, got {type(data)}')
                adapter = TypeAdapter(AllTinyMessages)
                return adapter.validate_python(data)

            case _:
                raise ValueError(
                    f'Unsupported data type: {type_}, supported types: {", ".join(DataType.__members__)}'
                )

    @classmethod
    def from_record(cls, record: dict) -> TinyEventNode:
        return TinyEventNode(
            uuid=record['uuid'],
            subgraph_id=record['subgraph_id'],
            name=record['name'],
            description=record['description'],
            created_at=parse_db_date(record['created_at']),
            valid_at=parse_db_date(record['valid_at']),
            data_type=DataType(record['data_type']),
            data=cls._parse_data(record['data'], record['data_type']),
        )

    async def save(self, driver: BaseDriver) -> str:
        from tiny_graph.graph.multi_layer_graph.queries.node_queries import (
            create_event_node,
        )

        args = {
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'subgraph_id': self.subgraph_id,
            'created_at': self.created_at,
            'valid_at': self.valid_at,
            'data': self.serialized_data,
            'data_type': self.data_type.value,
        }

        return await driver.execute_query(
            query=create_event_node(driver.provider),
            **args,
        )


class TinyEntityNode(TinyNode):
    type = NodeType.ENTITY

    summary: str

    labels: list[str]

    name_embedding: list[float] | None = None

    @classmethod
    def from_record(cls, record: dict) -> TinyEntityNode:
        return TinyEntityNode(
            uuid=record['uuid'],
            subgraph_id=record['subgraph_id'],
            name=record['name'],
            created_at=parse_db_date(record['created_at']),
            summary=record['summary'],
            labels=record['labels'],
            name_embedding=record.get('name_embedding', None),
        )

    async def save(self, driver: BaseDriver) -> str:
        from tiny_graph.graph.multi_layer_graph.queries.node_queries import (
            create_entity_node,
        )

        args = {
            'uuid': self.uuid,
            'name': self.name,
            'subgraph_id': self.subgraph_id,
            'created_at': self.created_at,
            'summary': self.summary,
            'labels': self.labels,
            'name_embedding': self.name_embedding,
        }

        return await driver.execute_query(
            query=create_entity_node(driver.provider),
            **args,
        )

    async def embed(self, embedder: AbstractEmbedder) -> list[float]:
        self.name_embedding = await embedder.aembed(self.name)
        return self.name_embedding

    def __repr__(self) -> str:
        return compact_model_repr(self)


class TinyClusterNode(TinyNode):
    type = NodeType.CLUSTER

    summary: str

    name_embedding: list[float] | None = None

    @classmethod
    def from_record(cls, record: dict) -> TinyClusterNode:
        return TinyClusterNode(
            uuid=record['uuid'],
            subgraph_id=record['subgraph_id'],
            name=record['name'],
            created_at=parse_db_date(record['created_at']),
            summary=record['summary'],
            name_embedding=record.get('name_embedding', None),
        )

    async def save(self, driver: BaseDriver) -> str:
        from tiny_graph.graph.multi_layer_graph.queries.node_queries import (
            create_cluster_node,
        )

        args = {
            'uuid': self.uuid,
            'name': self.name,
            'subgraph_id': self.subgraph_id,
            'created_at': self.created_at,
            'summary': self.summary,
            'name_embedding': self.name_embedding,
        }

        return await driver.execute_query(
            query=create_cluster_node(driver.provider),
            **args,
        )

    @classmethod
    async def find_by_entity(
        cls, driver: BaseDriver, entity_uuid: str
    ) -> list[TinyClusterNode]:
        from tiny_graph.graph.multi_layer_graph.queries.cluster_queries import (
            find_entity_clusters,
        )

        results, _, _ = await driver.execute_query(
            query=find_entity_clusters(driver.provider),
            entity_uuid=entity_uuid,
        )

        return [TinyClusterNode.from_record(r) for r in results]
