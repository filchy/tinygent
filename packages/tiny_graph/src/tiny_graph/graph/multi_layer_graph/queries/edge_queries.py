from tiny_graph.graph.multi_layer_graph.types import EdgeType
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.types.provider import GraphProvider


def create_event_edge(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            MATCH (event:{NodeType.EVENT.value} {{ uuid: $event_node_uuid }})
            MATCH (entity:{NodeType.ENTITY.value} {{ uuid: $entity_node_uuid }})
            MERGE (event)-[e:{EdgeType.MENTIONS.value} {{ uuid: $uuid }}]->(node)
            SET e = {{
                uuid: $uuid,
                subgraph_id: $subgraph_id,
                created_at: $created_at,
                source_node_uuid: $cluster_node_uuid,
                target_node_uuid: $entity_node_uuid,
                name: {EdgeType.MENTIONS.value}
            }}
            RETURN e.uuid AS uuid
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def create_cluster_edge(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            MATCH (cluster:{NodeType.CLUSTER.value} {{ uuid: $cluster_node_uuid }})
            MATCH (entity:{NodeType.ENTITY.value} {{ uuid: $entity_node_uuid }})
            MERGE (cluster)-[e:{EdgeType.HAS_MEMBER.value} {{ uuid: $uuid }}]->(node)
            SET e = {{
                uuid: $uuid,
                subgraph_id: $subgraph_id,
                created_at: $created_at,
                source_node_uuid: $cluster_node_uuid,
                target_node_uuid: $entity_node_uuid,
                name: {EdgeType.HAS_MEMBER.value}
            }}
            RETURN e.uuid AS uuid
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def create_entity_edge(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            MATCH (source:{NodeType.ENTITY.value} {{ uuid: $source_node_uuid }})
            MATCH (target:{NodeType.ENTITY.value} {{ uuid: $target_node_uuid }})
            MERGE (source)-[e:RELATES_TO {{ uuid: $edge_uuid}}]->(target)
            SET e = {{
                uuid: $edge_uuid,
                subgraph_id: $subgraph_id,
                source_node_uuid: $source_node_uuid,
                target_node_uuid: $target_node_uuid,
                created_at: $created_at,
                name: $name,
                fact: $fact,
                fact_embedding: $fact_embedding,
                events: $events,
                expired_at: $expired_at,
                valid_at: $valid_at,
                invalid_at: $invalid_at,
                attributes: $attributes
            }}
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def find_entity_edge_by_targets(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            MATCH (source:{NodeType.ENTITY.value} {{ uuid: $source_uuid }})-[e:RELATES_TO]->(target:{NodeType.ENTITY.value} {{ uuid: $target_uuid }})
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

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def save_entity_edges_bulk(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            UNWIND $edges AS edge
            MATCH (source:{NodeType.ENTITY.value} {{ uuid: edge.source_node_uuid }})
            MATCH (target:{NodeType.ENTITY.value} {{ uuid: edge.target_node_uuid }})
            MERGE (source)-[e:RELATES_TO {{ uuid: edge.uuid }}]->(target)
            SET e = {{
                uuid: edge.uuid,
                subgraph_id: edge.subgraph_id,
                source_node_uuid: edge.source_node_uuid,
                target_node_uuid: edge.target_node_uuid,
                created_at: edge.created_at,
                name: edge.name,
                fact: edge.fact,
                fact_embedding: edge.fact_embedding,
                events: edge.events,
                expired_at: edge.expired_at,
                valid_at: edge.valid_at,
                invalid_at: edge.invalid_at,
                attributes: edge.attributes
            }}
            RETURN collect(e.uuid) AS uuids
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def save_event_edges_bulk(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            UNWIND $edges AS edge
            MATCH (event:{NodeType.EVENT.value} {{ uuid: edge.event_node_uuid }})
            MATCH (entity:{NodeType.ENTITY.value} {{ uuid: edge.entity_node_uuid }})
            MERGE (event)-[e:{EdgeType.MENTIONS.value} {{ uuid: edge.uuid }}]->(entity)
            SET e = {{
                uuid: edge.uuid,
                subgraph_id: edge.subgraph_id,
                created_at: edge.created_at,
                source_node_uuid: edge.event_node_uuid,
                target_node_uuid: edge.entity_node_uuid,
                name: '{EdgeType.MENTIONS.value}'
            }}
            RETURN collect(e.uuid) AS uuids
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def save_cluster_edges_bulk(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
            UNWIND $edges AS edge
            MATCH (cluster:{NodeType.CLUSTER.value} {{ uuid: edge.cluster_node_uuid }})
            MATCH (entity:{NodeType.ENTITY.value} {{ uuid: edge.entity_node_uuid }})
            MERGE (cluster)-[e:{EdgeType.HAS_MEMBER.value} {{ uuid: edge.uuid }}]->(entity)
            SET e = {{
                uuid: edge.uuid,
                subgraph_id: edge.subgraph_id,
                created_at: edge.created_at,
                source_node_uuid: edge.cluster_node_uuid,
                target_node_uuid: edge.entity_node_uuid,
                name: '{EdgeType.HAS_MEMBER.value}'
            }}
            RETURN collect(e.uuid) AS uuids
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )
