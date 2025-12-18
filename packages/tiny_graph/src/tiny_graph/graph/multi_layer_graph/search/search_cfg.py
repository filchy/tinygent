from enum import Enum
from pydantic import Field
from tinygent.types.base import TinyModel

from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEventNode


class EntitySearchMethods(Enum):
    COSINE_SIM = 'cosine_similarity'
    BM_25 = 'bm_25'


class EntityReranker(Enum):
    CROSS_ENCODER = 'cross_encoder'


class TinyEntitySearchConfig(TinyModel):
    search_methods: list[EntitySearchMethods] = Field(default=[EntitySearchMethods.COSINE_SIM])
    reranker: EntityReranker = Field(default=EntityReranker.CROSS_ENCODER)


class TinySearchConfig(TinyModel):
    limit: int = Field(default=5)

    entity_search: TinyEntitySearchConfig = Field(default_factory=TinyEntitySearchConfig)


class TinySearchResult(TinyModel):
    events: list[TinyEventNode] = Field(default_factory=list)
    entities: list[TinyEntityNode] = Field(default_factory=list)
    clusters: list[TinyClusterNode] = Field(default_factory=list)
