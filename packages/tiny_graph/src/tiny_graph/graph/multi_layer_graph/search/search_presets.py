from tiny_graph.graph.multi_layer_graph.search.search_cfg import EdgeReranker
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EdgeSearchMethods
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EntityReranker
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EntitySearchMethods
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinyEdgeSearchConfig
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinyEntitySearchConfig
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchConfig

NODE_HYBRID_SEARCH_RRF = TinySearchConfig(
    entity_search=TinyEntitySearchConfig(
        search_methods=[EntitySearchMethods.BM_25, EntitySearchMethods.COSINE_SIM],
        reranker=EntityReranker.RRF,
    )
)

EDGE_HYBRID_SEARCH_RRF = TinySearchConfig(
    edge_search=TinyEdgeSearchConfig(
        search_methods=[EdgeSearchMethods.BM_25, EdgeSearchMethods.COSINE_SIM],
        reranker=EdgeReranker.RRF,
    )
)
