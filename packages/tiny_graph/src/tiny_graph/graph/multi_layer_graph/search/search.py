import logging
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

logger = logging.getLogger(__name__)


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
        ((entity_nodes, entity_reranker_scores), )
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
        entities=entity_nodes,
        entity_reranker_scores=entity_reranker_scores,
    )


async def entity_search(
    clients: TinyGraphClients,
    query: str,
    query_vector: list[float] | None,
    *,
    limit: int,
    config: TinyEntitySearchConfig,
    subgraph_ids: list[str] | None,
) -> tuple[list[TinyEntityNode], list[float]]:
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
    entity_uuid_map = {e.uuid: e for single_group in searched_entities for e in single_group}

    reranked_uuids: list[str] = []
    reranked_scores: list[float] = []

    if config.reranker == EntityReranker.CROSS_ENCODER:
        entity_name_2_uuid_map = {
            e.name: e.uuid
            for single_group in searched_entities
            for e in single_group
        }
        reranked_results = await clients.cross_encoder.rank(
            query,
            list(entity_name_2_uuid_map.keys())
        )
        reranked_uuids = [
            entity_name_2_uuid_map[r[0][1]]
            for r in reranked_results
        ]
        reranked_scores = [
            r[1]
            for r in reranked_results
        ]

    reranked_entities = [entity_uuid_map[uuid] for uuid in reranked_uuids]
    return reranked_entities[:limit], reranked_scores[:limit]
