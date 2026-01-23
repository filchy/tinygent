from tiny_graph.graph.multi_layer_graph.edges import TinyEntityEdge
from tinygent.core.datamodels.embedder import AbstractEmbedder


async def entity_edge_batch_embeddings(
    embedder: AbstractEmbedder, edges: list[TinyEntityEdge]
) -> list[TinyEntityEdge]:
    if not edges:
        return []

    embeddings = await embedder.aembed_batch([e.fact for e in edges])
    for edge, emb in zip(edges, embeddings, strict=True):
        edge.fact_embedding = emb
    return edges
