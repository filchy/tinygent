from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tinygent.core.datamodels.embedder import AbstractEmbedder


async def cluster_node_batch_embeddings(
    embedder: AbstractEmbedder, clusters: list[TinyClusterNode]
) -> list[TinyClusterNode]:
    if not clusters:
        return []

    embeddings = await embedder.aembed_batch([c.name for c in clusters])
    for cluster, emb in zip(clusters, embeddings, strict=True):
        cluster.name_embedding = emb
    return clusters


async def entity_node_batch_embeddings(
    embedder: AbstractEmbedder, entities: list[TinyEntityNode]
) -> list[TinyEntityNode]:
    if not entities:
        return []

    embeddings = await embedder.aembed_batch([e.name for e in entities])
    for entity, emb in zip(entities, embeddings, strict=True):
        entity.name_embedding = emb
    return entities
