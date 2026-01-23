from __future__ import annotations

from datetime import datetime

from pydantic import Field

from tiny_graph.driver.base import BaseDriver
from tiny_graph.edge import TinyEdge
from tiny_graph.graph.multi_layer_graph.types import EdgeType
from tiny_graph.graph.multi_layer_graph.utils.model_repr import compact_model_repr
from tiny_graph.helper import parse_db_date
from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.utils.yaml import json


class TinyEntityEdge(TinyEdge):
    fact: str = Field(
        description='fact representing the edge and nodes that it connects'
    )

    fact_embedding: list[float] | None = Field(
        default=None, description='embedding of the fact'
    )

    events: list[str] = Field(
        default=[],
        description='list of episode ids that reference these entity edges',
    )

    expired_at: datetime | None = Field(
        default=None, description='datetime of when the node was invalidated'
    )

    valid_at: datetime | None = Field(
        default=None, description='datetime of when the fact became true'
    )

    invalid_at: datetime | None = Field(
        default=None, description='datetime of when the fact stopped being true'
    )

    async def save(self, driver: BaseDriver) -> str:
        from tiny_graph.graph.multi_layer_graph.queries.edge_queries import (
            create_entity_edge,
        )

        args = {
            'edge_uuid': self.uuid,
            'subgraph_id': self.subgraph_id,
            'source_node_uuid': self.source_node_uuid,
            'target_node_uuid': self.target_node_uuid,
            'created_at': self.created_at,
            'name': self.name,
            'fact': self.fact,
            'fact_embedding': self.fact_embedding,
            'events': self.events,
            'expired_at': self.expired_at,
            'valid_at': self.valid_at,
            'invalid_at': self.invalid_at,
            'attributes': json.dumps(self.attributes),
        }

        return await driver.execute_query(
            query=create_entity_edge(driver.provider),
            **args,
        )

    async def embed(self, embedder: AbstractEmbedder) -> list[float]:
        self.fact_embedding = await embedder.aembed(self.fact)
        return self.fact_embedding

    @classmethod
    def from_record(cls, record: dict) -> TinyEntityEdge:
        return TinyEntityEdge(
            uuid=record['uuid'],
            subgraph_id=record['subgraph_id'],
            source_node_uuid=record['source_node_uuid'],
            target_node_uuid=record['target_node_uuid'],
            created_at=parse_db_date(record['created_at']),
            name=record['name'],
            fact=record['fact'],
            fact_embedding=record.get('fact_embedding'),
            events=record.get('events', []),
            expired_at=(
                parse_db_date(record['expired_at']) if record.get('expired_at') else None
            ),
            valid_at=(
                parse_db_date(record['valid_at']) if record.get('valid_at') else None
            ),
            invalid_at=(
                parse_db_date(record['invalid_at']) if record.get('invalid_at') else None
            ),
            attributes=(
                json.loads(record['attributes'])
                if isinstance(record.get('attributes'), str)
                else record.get('attributes', {})
            ),
        )

    @classmethod
    async def find_by_targets(
        cls, driver: BaseDriver, source_uuid: str, target_uuid: str
    ) -> list[TinyEntityEdge]:
        from tiny_graph.graph.multi_layer_graph.queries.edge_queries import (
            find_entity_edge_by_targets,
        )

        query = find_entity_edge_by_targets(driver.provider)

        results, _, _ = await driver.execute_query(
            query,
            source_uuid=source_uuid,
            target_uuid=target_uuid,
        )
        return [TinyEntityEdge.from_record(r) for r in results]

    def __repr__(self) -> str:
        return compact_model_repr(self)


class TinyClusterEdge(TinyEdge):
    name: str = Field(default=EdgeType.HAS_MEMBER.value, frozen=True)

    async def save(self, driver: BaseDriver) -> str:
        from tiny_graph.graph.multi_layer_graph.queries.edge_queries import (
            create_cluster_edge,
        )

        args = {
            'uuid': self.uuid,
            'subgraph_id': self.subgraph_id,
            'created_at': self.created_at,
            'cluster_node_uuid': self.source_node_uuid,
            'entity_node_uuid': self.target_node_uuid,
        }

        return await driver.execute_query(
            query=create_cluster_edge(driver.provider),
            **args,
        )

    @classmethod
    def from_record(cls, record: dict) -> TinyClusterEdge:
        return TinyClusterEdge(
            uuid=record['uuid'],
            subgraph_id=record['subgraph_id'],
            source_node_uuid=record['source_node_uuid'],
            target_node_uuid=record['target_node_uuid'],
            created_at=parse_db_date(record['created_at']),
            name=record['name'],
        )


class TinyEventEdge(TinyEdge):
    name: str = Field(default=EdgeType.MENTIONS.value, frozen=True)

    async def save(self, driver: BaseDriver) -> str:
        from tiny_graph.graph.multi_layer_graph.queries.edge_queries import (
            create_event_edge,
        )

        args = {
            'uuid': self.uuid,
            'subgraph_id': self.subgraph_id,
            'created_at': self.created_at,
            'event_node_uuid': self.source_node_uuid,
            'entity_node_uuid': self.target_node_uuid,
        }

        return await driver.execute_query(
            query=create_event_edge(driver.provider),
            **args,
        )

    @classmethod
    def from_record(cls, record: dict) -> TinyClusterEdge:
        return TinyClusterEdge(
            uuid=record['uuid'],
            subgraph_id=record['subgraph_id'],
            source_node_uuid=record['source_node_uuid'],
            target_node_uuid=record['target_node_uuid'],
            created_at=parse_db_date(record['created_at']),
            name=record['name'],
        )
