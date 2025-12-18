from tinygent.runtime.executors import run_in_semaphore

from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EntityReranker
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EntitySearchMethods
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchResult
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinyEntitySearchConfig
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchConfig
from tiny_graph.graph.multi_layer_graph.search.search_utils import entity_fulltext_search
from tiny_graph.graph.multi_layer_graph.search.search_utils import entity_similarity_search


async def search(
    query: str,
    subgraph_ids: list[str] | None = None,
    *,
    clients: TinyGraphClients,
    config: TinySearchConfig = TinySearchConfig(),
) -> TinySearchResult:
    query_vector: list[float] | None = None
    if EntitySearchMethods.COSINE_SIM in config.entity_search.search_methods:
        query_vector = clients.embedder.embed(query)

    (
        (entities,)
    ) = await run_in_semaphore(
        entity_search(
            clients,
            query,
            query_vector,
            config=config.entity_search,
            subgraph_ids=subgraph_ids,
            limit=config.limit,
        )
    )

    return TinySearchResult(
        entities=entities,
    )


async def entity_search(
    clients: TinyGraphClients,
    query: str,
    query_vector: list[float] | None,
    *,
    limit: int,
    config: TinyEntitySearchConfig,
    subgraph_ids: list[str] | None,
) -> list[TinyEntityNode]:
    tasks = []
    searched_entities: list[list[TinyEntityNode]] = []

    # search stage
    if EntitySearchMethods.BM_25 in config.search_methods:
        tasks.append(
            entity_fulltext_search(clients, query, subgraph_ids=subgraph_ids, limit=limit)
        )

    if query_vector and EntitySearchMethods.COSINE_SIM in config.search_methods:
        tasks.append(
            entity_similarity_search(clients, query_vector, subgraph_ids=subgraph_ids, limit=limit)
        )

    if tasks:
        searched_entities = await run_in_semaphore(*tasks)

    # reranking stage
    if config.reranker == EntityReranker.CROSS_ENCODER:
        pass

    return [e for single_group in searched_entities for e in single_group]
