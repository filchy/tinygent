from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.types.provider import GraphProvider


def create_entity_node(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f'''
            MERGE (e:{NodeType.ENTITY.value} {{ uuid: $uuid }})
            SET e = {{
                uuid: $uuid,
                name: $name,
                subgraph_id: $subgraph_id,
                created_at: $created_at,
                summary: $summary,
                labels: $labels
            }}
            WITH e
            WHERE $name_embedding IS NOT NULL
            CALL db.create.setNodeVectorProperty(
                e,
                "name_embedding",
                $name_embedding
            )
            RETURN e.uuid AS uuid
        '''

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def create_event_node(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f'''
            MERGE (e:{NodeType.EVENT.value} {{ uuid: $uuid }})
            SET e = {{
                uuid: $uuid,
                name: $name,
                description: $description,
                subgraph_id: $subgraph_id,
                valid_at: $valid_at,
                created_at: $created_at,
                data: $data,
                data_type: $data_type
            }}
            return e.uuid as uuid
        '''

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def create_cluster_node(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f'''
            MERGE (e:{NodeType.CLUSTER.value} {{ uuid: $uuid }})
            SET e = {{
                uuid: $uuid,
                name: $name,
                subgraph_id: $subgraph_id,
                created_at: $created_at,
                summary: $summary,
            }}
            WITH e
            WHERE $name_embedding IS NOT NULL
            CALL db.create.setNodeVectorProperty(
                e,
                "name_embedding",
                $name_embedding
            )
            RETURN e.uuid AS uuid
        '''

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def get_last_n_event_nodes(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f'''
            MATCH (e:{NodeType.EVENT.value})
            WHERE ($subgraph_ids IS NULL OR size($subgraph_ids) = 0 OR e.subgraph_id in $subgraph_ids)
                AND e.valid_at <= $reference_time
            ORDER BY e.valid_at DESC, e.uuid
            LIMIT $last_n
            RETURN
                e.uuid        AS uuid,
                e.subgraph_id AS subgraph_id,
                e.name        AS name,
                e.description AS description,
                e.created_at  AS created_at,
                e.valid_at    AS valid_at,
                e.data        AS data,
                e.data_type   AS data_type
        '''

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )
