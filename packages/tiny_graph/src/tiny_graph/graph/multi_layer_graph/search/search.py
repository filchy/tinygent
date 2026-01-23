from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.edges import TinyEntityEdge
from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.search.search_cfg import ClusterSearchMethods
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EdgeReranker
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EdgeSearchMethods
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EntityReranker
from tiny_graph.graph.multi_layer_graph.search.search_cfg import EntitySearchMethods
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinyClusterSearchConfig
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinyEdgeSearchConfig
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinyEntitySearchConfig
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchConfig
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchFilters
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchResult
from tiny_graph.graph.multi_layer_graph.search.search_ranker import (
    rerank_candidates_cross_encoder,
)
from tiny_graph.graph.multi_layer_graph.search.search_ranker import rerank_candidates_rrf
from tiny_graph.graph.multi_layer_graph.search.search_utils import (
    cluster_fulltext_search,
)
from tiny_graph.graph.multi_layer_graph.search.search_utils import (
    cluster_similarity_search,
)
from tiny_graph.graph.multi_layer_graph.search.search_utils import edge_fulltext_search
from tiny_graph.graph.multi_layer_graph.search.search_utils import edge_similarity_search
from tiny_graph.graph.multi_layer_graph.search.search_utils import entity_fulltext_search
from tiny_graph.graph.multi_layer_graph.search.search_utils import (
    entity_similarity_search,
)
from tinygent.core.runtime.executors import run_in_semaphore


async def search(
    query: str,
    subgraph_ids: list[str] | None = None,
    *,
    clients: TinyGraphClients,
    config: TinySearchConfig = TinySearchConfig(),
    filters: TinySearchFilters | None = None,
) -> TinySearchResult:
    query_vector: list[float] | None = None
    if (
        config.entity_search
        and EntitySearchMethods.COSINE_SIM in config.entity_search.search_methods
    ):
        query_vector = clients.embedder.embed(query)

    (
        (
            (edges, edge_reranker_scores),
            (entities, entity_reranker_scores),
            (clusters, cluster_reranker_scores),
        )
    ) = await run_in_semaphore(
        edge_search(
            clients,
            query,
            query_vector,
            None,
            None,
            config=config.edge_search,
            filters=filters,
            subgraph_ids=subgraph_ids,
            limit=config.limit,
        ),
        entity_search(
            clients,
            query,
            query_vector,
            config=config.entity_search,
            filters=filters,
            subgraph_ids=subgraph_ids,
            limit=config.limit,
        ),
        cluster_search(
            clients,
            query,
            query_vector,
            config=config.cluster_search,
            filters=filters,
            subgraph_ids=subgraph_ids,
            limit=config.limit,
        ),
    )

    return TinySearchResult(
        entities=entities,
        entity_reranker_scores=entity_reranker_scores,
        edges=edges,
        edge_reranker_scores=edge_reranker_scores,
        clusters=clusters,
        cluster_reranker_scores=cluster_reranker_scores,
    )


async def cluster_search(
    clients: TinyGraphClients,
    query: str,
    query_vector: list[float] | None,
    *,
    limit: int,
    config: TinyClusterSearchConfig | None,
    filters: TinySearchFilters | None = None,
    subgraph_ids: list[str] | None,
) -> tuple[list[TinyClusterNode], list[float]]:
    if not config:
        return [], []

    tasks = []
    search_results: list[list[TinyClusterNode]] = []

    # search stage
    if ClusterSearchMethods.BM_25 in config.search_methods:
        tasks.append(
            cluster_fulltext_search(
                clients, query, subgraph_ids=subgraph_ids, filters=filters, limit=limit
            )
        )

    if query_vector and ClusterSearchMethods.COSINE_SIM in config.search_methods:
        tasks.append(
            cluster_similarity_search(
                clients,
                query_vector,
                subgraph_ids=subgraph_ids,
                filters=filters,
                limit=limit,
            )
        )

    if tasks:
        search_results = await run_in_semaphore(*tasks)

    # reranking stage
    cluster_uuid_map = {
        c.uuid: c for single_group in search_results for c in single_group
    }

    reranked_uuids: list[str] = []
    reranked_scores: list[float] = []

    if config.reranker == EdgeReranker.RRF:
        reranked_uuids, reranked_scores = rerank_candidates_rrf(search_results)

    elif config.reranker == EdgeReranker.CROSS_ENCODER:
        reranked_uuids, reranked_scores = await rerank_candidates_cross_encoder(
            query, search_results, clients.cross_encoder
        )

    reranked_clusters = [cluster_uuid_map[uuid] for uuid in reranked_uuids]
    return reranked_clusters[:limit], reranked_scores[:limit]


async def entity_search(
    clients: TinyGraphClients,
    query: str,
    query_vector: list[float] | None,
    *,
    limit: int,
    config: TinyEntitySearchConfig | None,
    filters: TinySearchFilters | None = None,
    subgraph_ids: list[str] | None,
) -> tuple[list[TinyEntityNode], list[float]]:
    if not config:
        return [], []

    tasks = []
    search_results: list[list[TinyEntityNode]] = []

    # search stage
    if EntitySearchMethods.BM_25 in config.search_methods:
        tasks.append(
            entity_fulltext_search(
                clients, query, subgraph_ids=subgraph_ids, filters=filters, limit=limit
            )
        )

    if query_vector and EntitySearchMethods.COSINE_SIM in config.search_methods:
        tasks.append(
            entity_similarity_search(
                clients,
                query_vector,
                subgraph_ids=subgraph_ids,
                filters=filters,
                limit=limit,
            )
        )

    if tasks:
        search_results = await run_in_semaphore(*tasks)

    # reranking stage
    entity_uuid_map = {
        e.uuid: e for single_group in search_results for e in single_group
    }

    reranked_uuids: list[str] = []
    reranked_scores: list[float] = []

    if config.reranker == EntityReranker.RRF:
        reranked_uuids, reranked_scores = rerank_candidates_rrf(search_results)

    elif config.reranker == EntityReranker.CROSS_ENCODER:
        reranked_uuids, reranked_scores = await rerank_candidates_cross_encoder(
            query, search_results, clients.cross_encoder
        )

    reranked_entities = [entity_uuid_map[uuid] for uuid in reranked_uuids]
    return reranked_entities[:limit], reranked_scores[:limit]


async def edge_search(
    clients: TinyGraphClients,
    query: str,
    query_vector: list[float] | None,
    source_node_uuid: str | None,
    target_node_uuid: str | None,
    *,
    limit: int,
    config: TinyEdgeSearchConfig | None,
    filters: TinySearchFilters | None = None,
    subgraph_ids: list[str] | None = None,
) -> tuple[list[TinyEntityEdge], list[float]]:
    if not config:
        return [], []

    tasks = []
    search_results: list[list[TinyEntityEdge]] = []

    if EdgeSearchMethods.BM_25 in config.search_methods:
        tasks.append(
            edge_fulltext_search(
                clients,
                query,
                source_node_uuid,
                target_node_uuid,
                subgraph_ids=subgraph_ids,
                filters=filters,
                limit=limit,
            )
        )

    if query_vector and EdgeSearchMethods.COSINE_SIM in config.search_methods:
        tasks.append(
            edge_similarity_search(
                clients,
                query_vector,
                source_node_uuid,
                target_node_uuid,
                subgraph_ids=subgraph_ids,
                filters=filters,
                limit=limit,
            )
        )

    if tasks:
        search_results = await run_in_semaphore(*tasks)

    # reranking stage
    edge_uuid_map = {e.uuid: e for single_group in search_results for e in single_group}

    reranked_uuids: list[str] = []
    reranked_scores: list[float] = []

    if config.reranker == EdgeReranker.RRF:
        reranked_uuids, reranked_scores = rerank_candidates_rrf(search_results)

    elif config.reranker == EdgeReranker.CROSS_ENCODER:
        reranked_uuids, reranked_scores = await rerank_candidates_cross_encoder(
            query, search_results, clients.cross_encoder
        )

    reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]
    return reranked_edges[:limit], reranked_scores[:limit]
