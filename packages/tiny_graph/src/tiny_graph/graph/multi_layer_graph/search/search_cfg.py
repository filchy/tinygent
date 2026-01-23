from enum import Enum

from pydantic import Field

from tiny_graph.graph.multi_layer_graph.edges import TinyEntityEdge
from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEventNode
from tiny_graph.types.provider import GraphProvider
from tinygent.core.types.base import TinyModel


class EdgeSearchMethods(Enum):
    COSINE_SIM = 'cosine_similarity'
    BM_25 = 'bm_25'


class EntitySearchMethods(Enum):
    COSINE_SIM = 'cosine_similarity'
    BM_25 = 'bm_25'


class ClusterSearchMethods(Enum):
    COSINE_SIM = 'cosine_similarity'
    BM_25 = 'bm_25'


class EdgeReranker(Enum):
    RRF = 'rrf'
    CROSS_ENCODER = 'cross_encoder'


class EntityReranker(Enum):
    RRF = 'rrf'
    CROSS_ENCODER = 'cross_encoder'


class ClusterReranker(Enum):
    RRF = 'rrf'
    CROSS_ENCODER = 'cross_encoder'


class TinySearchFilters(TinyModel):
    entity_uuids: list[str] | None = None
    edge_uuids: list[str] | None = None
    cluster_uuids: list[str] | None = None

    def build_query(self, provider: GraphProvider, *use_fields: str) -> str:
        clauses: list[str] = []

        if 'entity_uuids' in use_fields and self.entity_uuids:
            clauses.append(self._in_clause(provider, 'e.uuid', 'entity_uuids'))

        if 'edge_uuids' in use_fields and self.edge_uuids:
            clauses.append(self._in_clause(provider, 'e.uuid', 'edge_uuids'))

        if 'cluster_uuids' in use_fields and self.edge_uuids:
            clauses.append(self._in_clause(provider, 'c.uuid', 'cluster_uuids'))

        if not clauses:
            return ''

        return 'AND ' + '\nAND '.join(clauses)

    def _in_clause(self, provider: GraphProvider, field: str, param: str) -> str:
        match provider:
            case GraphProvider.NEO4J:
                return f"""
                (
                    ${param} IS NULL
                    OR size(${param}) = 0
                    OR {field} IN ${param}
                )
                """
            case _:
                raise NotImplementedError(f'Provider not supported: {provider}')


class TinyEntitySearchConfig(TinyModel):
    search_methods: list[EntitySearchMethods] = Field(
        default=[EntitySearchMethods.COSINE_SIM]
    )
    reranker: EntityReranker = Field(default=EntityReranker.CROSS_ENCODER)


class TinyEdgeSearchConfig(TinyModel):
    search_methods: list[EdgeSearchMethods] = Field(
        default=[EdgeSearchMethods.COSINE_SIM]
    )
    reranker: EdgeReranker = Field(default=EdgeReranker.CROSS_ENCODER)


class TinyClusterSearchConfig(TinyModel):
    search_methods: list[ClusterSearchMethods] = Field(
        default=[ClusterSearchMethods.COSINE_SIM]
    )
    reranker: ClusterReranker = Field(default=ClusterReranker.CROSS_ENCODER)


class TinySearchConfig(TinyModel):
    limit: int = Field(default=5)

    entity_search: TinyEntitySearchConfig | None = Field(default=None)
    edge_search: TinyEdgeSearchConfig | None = Field(default=None)
    cluster_search: TinyClusterSearchConfig | None = Field(default=None)


class TinySearchResult(TinyModel):
    events: list[TinyEventNode] = Field(default_factory=list)
    event_reranker_scores: list[float] = Field(default_factory=list)

    entities: list[TinyEntityNode] = Field(default_factory=list)
    entity_reranker_scores: list[float] = Field(default_factory=list)

    edges: list[TinyEntityEdge] = Field(default_factory=list)
    edge_reranker_scores: list[float] = Field(default_factory=list)

    clusters: list[TinyClusterNode] = Field(default_factory=list)
    cluster_reranker_scores: list[float] = Field(default_factory=list)
