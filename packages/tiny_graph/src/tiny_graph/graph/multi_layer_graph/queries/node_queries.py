from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.types.provider import GraphProvider


def save_cluster_nodes_bulk(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            UNWIND $clusters AS cluster
            MERGE (c:{NodeType.CLUSTER.value} {{ uuid: cluster.uuid }})
            SET c = {{
                uuid: cluster.uuid,
                name: cluster.name,
                subgraph_id: cluster.subgraph_id,
                created_at: cluster.created_at,
                summary: cluster.summary
            }}
            WITH c, cluster
            WHERE cluster.name_embedding IS NOT NULL
            CALL db.create.setNodeVectorProperty(
                c,
                "name_embedding",
                cluster.name_embedding
            )
            RETURN collect(c.uuid) AS uuids
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def save_event_nodes_bulk(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            UNWIND $events AS event
            MERGE (e:{NodeType.EVENT.value} {{ uuid: event.uuid }})
            SET e = {{
                uuid: event.uuid,
                name: event.name,
                description: event.description,
                subgraph_id: event.subgraph_id,
                created_at: event.created_at,
                valid_at: event.valid_at,
                data: event.data,
                data_type: event.data_type
            }}
            RETURN collect(e.uuid) AS uuids
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def save_entity_nodes_bulk(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            UNWIND $entities as entity
            MERGE (e:{NodeType.ENTITY.value} {{ uuid: entity.uuid }})
            SET e = {{
                uuid: entity.uuid,
                name: entity.name,
                subgraph_id: entity.subgraph_id,
                created_at: entity.created_at,
                summary: entity.summary,
                labels: entity.labels
            }}
            WITH e, entity
            WHERE entity.name_embedding IS NOT NULL
            CALL db.create.setNodeVectorProperty(
                e,
                "name_embedding",
                entity.name_embedding
            )
            RETURN collect(e.uuid) AS uuids
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def create_entity_node(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
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
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def create_event_node(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
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
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def create_cluster_node(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
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
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def get_last_n_event_nodes(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
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
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )
