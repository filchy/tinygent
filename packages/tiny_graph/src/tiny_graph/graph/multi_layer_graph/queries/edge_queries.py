from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.types.provider import GraphProvider


def create_entity_edge(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f'''
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
        '''
    
    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def find_entity_edge_by_targets(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f'''
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
        '''

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )
