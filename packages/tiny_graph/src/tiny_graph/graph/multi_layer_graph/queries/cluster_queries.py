from tiny_graph.graph.multi_layer_graph.types import EdgeType
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.types.provider import GraphProvider


def find_entity_clusters(provider: GraphProvider) -> str:
    if provider == GraphProvider.NEO4J:
        return f"""
        MATCH (c:{NodeType.CLUSTER.value})-[:{EdgeType.HAS_MEMBER.value}]->(n:{NodeType.ENTITY.value} {{ uuid: $entity_uuid }})
        RETURN
            c.uuid AS uuid,
            c.name AS name,
            c.subgraph_id AS subgraph_id,
            c.created_at AS created_at,
            c.name_embedding AS name_embedding,
            c.summary AS summary
        """

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )
