from collections import defaultdict

from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.edges import TinyEntityEdge
from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchFilters
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.types.provider import GraphProvider


def rrf(
    rank_uuid_table: list[list[str]], k: int = 1, min_score: float = 0.0
) -> tuple[list[str], list[float]]:
    def _check_min_score(x: tuple[str, float]) -> bool:
        return x[1] >= min_score

    scores: dict[str, float] = defaultdict(float)
    for ranking_method in rank_uuid_table:
        for index_in_ranking, uuid in enumerate(ranking_method):
            scores[uuid] += 1 / (k + index_in_ranking)

    score_list: list[tuple[str, float]] = list(scores.items())
    score_list.sort(reverse=True, key=lambda x: x[1])
    score_list = list(filter(_check_min_score, score_list))

    return (
        [s[0] for s in score_list],
        [s[1] for s in score_list],
    )


async def entity_similarity_search(
    clients: TinyGraphClients,
    query_vector: list[float],
    *,
    subgraph_ids: list[str] | None = None,
    filters: TinySearchFilters | None = None,
    limit: int = 5,
    min_score: float = 0.0,
) -> list[TinyEntityNode]:
    provider = clients.driver.provider

    if provider == GraphProvider.NEO4J:
        filter_clause = filters.build_query(provider, 'entity_uuids') if filters else ''
        query = f"""
            CALL db.index.vector.queryNodes(
                '{NodeType.ENTITY.value}_{clients.safe_embed_model}_name_embedding_index',
                $limit,
                $query_vector
            )
            YIELD node as e, score
            WHERE score >= $min_score
            AND (
                $subgraph_ids IS NULL
                OR size($subgraph_ids) = 0
                OR e.subgraph_id in $subgraph_ids
            )
            {filter_clause}

            RETURN
                e.uuid AS uuid,
                e.name AS name,
                e.subgraph_id AS subgraph_id,
                e.labels AS labels,
                e.created_at AS created_at,
                e.name_embedding AS name_embedding,
                e.summary as summary

            ORDER BY score DESC, e.uuid
        """

        results, _, _ = await clients.driver.execute_query(
            query,
            **{
                'query_vector': query_vector,
                'subgraph_ids': subgraph_ids or [],
                'entity_uuids': filters.entity_uuids if filters else None,
                'limit': limit,
                'min_score': min_score,
            },
        )

        return [TinyEntityNode.from_record(r) for r in results]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


async def entity_fulltext_search(
    clients: TinyGraphClients,
    query: str,
    *,
    subgraph_ids: list[str] | None = None,
    filters: TinySearchFilters | None = None,
    limit: int = 5,
) -> list[TinyEntityNode]:
    provider = clients.driver.provider

    if provider == GraphProvider.NEO4J:
        filter_clause = filters.build_query(provider, 'entity_uuids') if filters else ''
        q = f"""
            CALL db.index.fulltext.queryNodes(
                '{NodeType.ENTITY.value}_fulltext_index',
                $text_query,
                {{limit: $limit}}
            )
            YIELD node as e, score
            WHERE
                (
                    $subgraph_ids IS NULL
                    OR size($subgraph_ids) = 0
                    OR e.subgraph_id in $subgraph_ids
                )
            {filter_clause}

            RETURN
                e.uuid AS uuid,
                e.name AS name,
                e.subgraph_id AS subgraph_id,
                e.labels AS labels,
                e.created_at AS created_at,
                e.name_embedding AS name_embedding,
                e.summary as summary
            ORDER BY score DESC, e.uuid
        """
        results, _, _ = await clients.driver.execute_query(
            q,
            **{
                'text_query': query,
                'subgraph_ids': subgraph_ids,
                'entity_uuids': filters.entity_uuids if filters else None,
                'limit': limit,
            },
        )

        return [TinyEntityNode.from_record(r) for r in results]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


async def entity_bfs_search(
    clients: TinyGraphClients,
) -> list[TinyEntityNode]:
    # TODO: implement after edges will be implemented and created
    return []


async def edge_similarity_search(
    clients: TinyGraphClients,
    query_vector: list[float],
    source_node_uuid: str | None,
    target_node_uuid: str | None,
    *,
    subgraph_ids: list[str] | None = None,
    filters: TinySearchFilters | None = None,
    limit: int = 5,
    min_score: float = 0.0,
) -> list[TinyEntityEdge]:
    provider = clients.driver.provider

    if provider == GraphProvider.NEO4J:
        filter_clause = filters.build_query(provider, 'entity_uuids') if filters else ''
        q = f"""
        CALL db.index.vector.queryRelationships(
            'RELATES_TO_{clients.safe_embed_model}_fact_embedding_index',
            $limit,
            $query_vector
        )
        YIELD relationship as e, score
        WHERE score >= $min_score
        AND
            (
                $subgraph_ids IS NULL
                OR size($subgraph_ids) = 0
                OR e.subgraph_id in $subgraph_ids
            )
        AND
            (
                $source_node_uuid IS NULL
                OR e.source_node_uuid = $source_node_uuid
            )
        AND
            (
                $target_node_uuid IS NULL
                OR e.target_node_uuid = $target_node_uuid
            )
        {filter_clause}

        RETURN
            e.uuid AS uuid,
            e.subgraph_id AS subgraph_id,
            e.source_node_uuid AS source_node_uuid,
            e.target_node_uuid AS target_node_uuid,
            e.created_at AS created_at,
            e.name AS name,
            e.fact AS fact,
            e.fact_embedding AS fact_embedding,
            e.events AS events,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at,
            e.attributes AS attributes

        ORDER BY score DESC, e.uuid
        """

        results, _, _ = await clients.driver.execute_query(
            q,
            **{
                'query_vector': query_vector,
                'subgraph_ids': subgraph_ids,
                'limit': limit,
                'target_node_uuid': target_node_uuid,
                'source_node_uuid': source_node_uuid,
                'entity_uuids': filters.entity_uuids if filters else None,
                'min_score': min_score,
            },
        )

        return [TinyEntityEdge.from_record(r) for r in results]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


async def edge_fulltext_search(
    clients: TinyGraphClients,
    query: str,
    source_node_uuid: str | None,
    target_node_uuid: str | None,
    *,
    filters: TinySearchFilters | None = None,
    subgraph_ids: list[str] | None = None,
    limit: int = 5,
) -> list[TinyEntityEdge]:
    provider = clients.driver.provider

    if provider == GraphProvider.NEO4J:
        filter_clause = filters.build_query(provider, 'edge_uuids') if filters else ''
        q = f"""
        CALL db.index.fulltext.queryRelationships(
            'edge_name_and_fact',
            $text_query,
            {{limit: $limit}}
        )
        YIELD relationship as e, score
        WHERE
            (
                $subgraph_ids IS NULL
                OR size($subgraph_ids) = 0
                OR e.subgraph_id in $subgraph_ids
            )
        AND
            (
                $source_node_uuid IS NULL
                OR e.source_node_uuid = $source_node_uuid
            )
        AND
            (
                $target_node_uuid IS NULL
                OR e.target_node_uuid = $target_node_uuid
            )
        {filter_clause}

        RETURN
            e.uuid AS uuid,
            e.subgraph_id AS subgraph_id,
            e.source_node_uuid AS source_node_uuid,
            e.target_node_uuid AS target_node_uuid,
            e.created_at AS created_at,
            e.name AS name,
            e.fact AS fact,
            e.fact_embedding AS fact_embedding,
            e.events AS events,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at,
            e.attributes AS attributes
        """

        results, _, _ = await clients.driver.execute_query(
            q,
            **{
                'text_query': query,
                'subgraph_ids': subgraph_ids,
                'target_node_uuid': target_node_uuid,
                'source_node_uuid': source_node_uuid,
                'edge_uuids': filters.edge_uuids if filters else None,
                'limit': limit,
            },
        )

        return [TinyEntityEdge.from_record(r) for r in results]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


async def cluster_similarity_search(
    clients: TinyGraphClients,
    query_vector: list[float],
    *,
    subgraph_ids: list[str] | None = None,
    filters: TinySearchFilters | None = None,
    limit: int = 5,
    min_score: float = 0.0,
) -> list[TinyClusterNode]:
    provider = clients.driver.provider

    if provider == GraphProvider.NEO4J:
        filter_clause = filters.build_query(provider, 'cluster_uuids') if filters else ''

        query = f"""
            CALL db.index.vector.queryNodes(
                '{NodeType.CLUSTER.value}_{clients.safe_embed_model}_name_embedding_index',
                $limit,
                $query_vector
            )
            YIELD node as c, score
            WHERE score >= $min_score
            AND (
                $subgraph_ids IS NULL
                OR size($subgraph_ids) = 0
                OR c.subgraph_id in $subgraph_ids
            )
            {filter_clause}

            RETURN
                c.uuid AS uuid,
                c.name AS name,
                c.subgraph_id AS subgraph_id,
                c.created_at AS created_at,
                c.name_embedding AS name_embedding,
                c.summary as summary
            ORDER BY score DESC, c.uuid
        """

        results, _, _ = await clients.driver.execute_query(
            query,
            **{
                'query_vector': query_vector,
                'subgraph_ids': subgraph_ids or [],
                'cluster_uuids': filters.cluster_uuids if filters else None,
                'limit': limit,
                'min_score': min_score,
            },
        )

        return [TinyClusterNode.from_record(r) for r in results]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


async def cluster_fulltext_search(
    clients: TinyGraphClients,
    query: str,
    subgraph_ids: list[str] | None = None,
    filters: TinySearchFilters | None = None,
    limit: int = 5,
) -> list[TinyClusterNode]:
    provider = clients.driver.provider

    if provider == GraphProvider.NEO4J:
        filter_clause = filters.build_query(provider, 'cluster_uuids') if filters else ''
        q = f"""
            CALL db.index.fulltext.queryNodes(
                '{NodeType.CLUSTER.value}_fulltext_index',
                $text_query,
                {{limit: $limit}}
            )
            YIELD node as c, score
            WHERE
                (
                    $subgraph_ids IS NULL
                    OR size($subgraph_ids) = 0
                    OR c.subgraph_id in $subgraph_ids
                )
            {filter_clause}

            RETURN
                c.uuid AS uuid,
                c.name AS name,
                c.subgraph_id AS subgraph_id,
                c.created_at AS created_at,
                c.name_embedding AS name_embedding,
                c.summary as summary
            ORDER BY score DESC, c.uuid
        """
        results, _, _ = await clients.driver.execute_query(
            q,
            **{
                'text_query': query,
                'subgraph_ids': subgraph_ids,
                'cluster_uuids': filters.cluster_uuids if filters else None,
                'limit': limit,
            },
        )

        return [TinyClusterNode.from_record(q) for q in results]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


if __name__ == '__main__':
    # docs: a, b, c
    rank_table = [['a', 'c', 'b'], ['c', 'b', 'a']]

    final_uuids, final_scores = rrf(rank_table, k=1)
    for i, (uuid, score) in enumerate(zip(final_uuids, final_scores, strict=True)):
        print(f'{i + 1}. {uuid} with score: {score}')
