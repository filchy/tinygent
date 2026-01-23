from collections.abc import Sequence

from tiny_graph.edge import TinyEdge
from tiny_graph.graph.multi_layer_graph.search.search_utils import rrf
from tiny_graph.node import TinyNode
from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoder


async def rerank_candidates_cross_encoder(
    query: str,
    candidates: Sequence[Sequence[TinyNode | TinyEdge]],
    cross_encoder: AbstractCrossEncoder,
) -> tuple[list[str], list[float]]:
    candidate_name_2_uuid_map = {
        c.name: c.uuid for single_group in candidates for c in single_group
    }
    reranked_results = await cross_encoder.rank(
        query, list(candidate_name_2_uuid_map.keys())
    )
    reranked_uuids = [candidate_name_2_uuid_map[r[0][1]] for r in reranked_results]
    reranked_scores = [r[1] for r in reranked_results]
    return reranked_uuids, reranked_scores


def rerank_candidates_rrf(
    candidates: Sequence[Sequence[TinyNode | TinyEdge]],
) -> tuple[list[str], list[float]]:
    ranking_table = [[c.uuid for c in method] for method in candidates]
    return rrf(ranking_table)
