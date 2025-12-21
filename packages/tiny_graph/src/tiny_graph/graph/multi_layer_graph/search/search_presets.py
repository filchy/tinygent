from tiny_graph.graph.multi_layer_graph.search.search_cfg import EntityReranker
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EntitySearchMethods
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinyEntitySearchConfig
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchConfig


NODE_HYBRID_SEARCH_CROSS_ENCODER = TinySearchConfig(
    entity_search=TinyEntitySearchConfig(
        search_methods=[EntitySearchMethods.BM_25, EntitySearchMethods.COSINE_SIM],
        reranker=EntityReranker.CROSS_ENCODER,
    )
)
