from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.types.provider import GraphProvider


async def entity_similarity_search(
    clients: TinyGraphClients,
    query_vector: list[float],
    *,
    subgraph_ids: list[str] | None = None,
    limit: int = 5,
    min_score: float = 0.0,
) -> list[TinyEntityNode]:
    provider = clients.driver.provider

    if provider == GraphProvider.NEO4J:
        query = f'''
            CALL db.index.vector.queryNodes(
                '{NodeType.ENTITY.value}_{clients.safe_embed_model}_name_embedding_index',
                $limit,
                $query_vector
            )
            YIELD node as e, score
            WHERE score > $min_score
            AND (
                $subgraph_ids IS NULL
                OR size($subgraph_ids) = 0
                OR e.subgraph_id in $subgraph_ids
            )
            RETURN
                e.uuid AS uuid,
                e.name AS name,
                e.subgraph_id AS subgraph_id,
                e.labels AS labels,
                e.created_at AS created_at,
                e.name_embedding AS name_embedding,
                e.summary as summary
            ORDER BY score DESC, e.uuid
        '''

        results, _, _ = await clients.driver.execute_query(query, **{
            'query_vector': query_vector,
            'subgraph_ids': subgraph_ids or [],
            'limit': limit,
            'min_score': min_score,
        })

        return [TinyEntityNode.from_record(r) for r in results]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


async def entity_fulltext_search(
    clients: TinyGraphClients,
    query: str,
    *,
    subgraph_ids: list[str] | None = None,
    limit: int = 5,
) -> list[TinyEntityNode]:
    provider = clients.driver.provider

    if provider == GraphProvider.NEO4J:
        q = f'''
            CALL db.index.fulltext.queryNodes(
                '{NodeType.ENTITY.value}_fulltext_index',
                $text_query,
                {{limit: $limit}}
            )
            YIELD node as e, score
            WHERE $subgraph_ids IS NULL
                OR size($subgraph_ids) = 0
                OR e.subgraph_id in $subgraph_ids

            RETURN
                e.uuid AS uuid,
                e.name AS name,
                e.subgraph_id AS subgraph_id,
                e.labels AS labels,
                e.created_at AS created_at,
                e.name_embedding AS name_embedding,
                e.summary as summary
            ORDER BY score DESC, e.uuid
        '''
        results, _, _ = await clients.driver.execute_query(q, **{
            'text_query': query,
            'subgraph_ids': subgraph_ids,
            'limit': limit,
        })

        return [TinyEntityNode.from_record(r) for r in results]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


async def entity_bfs_search(
    clients: TinyGraphClients,
) -> list[TinyEntityNode]:
    # TODO: implement after edges will be implemented and created
    return []
