from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import logging

from tinygent.datamodels.llm import AbstractLLM
from tinygent.runtime.executors import run_in_semaphore

from tiny_graph.driver.base import BaseDriver
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEventNode
from tiny_graph.graph.multi_layer_graph.queries.node_queries import get_last_n_event_nodes
from tiny_graph.graph.multi_layer_graph.search.search import search
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchResult
from tiny_graph.graph.multi_layer_graph.search.search_presets import NODE_HYBRID_SEARCH_CROSS_ENCODER
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import has_high_entropy, jaccard_similarity
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import normalize_string_for_fuzzy
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import lsh_bands
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import normalize_string_exact
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import minhash_signature
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import shingles
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import _TINY_FUZZY_JACCARD_THRESHOLD
from tiny_graph.types.provider import GraphProvider

logger = logging.getLogger(__name__)


@dataclass
class EntityCandidateIndex:
    entities_by_uuid: dict[str, TinyEntityNode]

    entities_by_norm_name: defaultdict[str, list[TinyEntityNode]]

    shingles_bu_uuid: dict[str, set[str]]

    lsh_by_uuid: defaultdict[tuple[int, tuple[int, ...]], list[str]]


@dataclass
class EntityDeduplicationState:
    """
    Deduplication result state.

    uuid_map mapping rules:
    - a -> b : extracted entity `a` was matched to an existing canonical entity `b`
    - a -> a : extracted entity `a` has NO match and remains canonical itself (new entity)
    - missing: never happens after finalization (all entities are explicitly mapped)
    """

    # Maps extracted entity UUID -> existing canonical entity UUID
    uuid_map: dict[str, str]

    resolved_entities: list[TinyEntityNode | None]

    unresolved_indices: list[int]

    duplicate_pairs: list[tuple[TinyEntityNode, TinyEntityNode]]


async def _find_entity_duplicite_candidates(
    clients: TinyGraphClients,
    extracted_entities: list[TinyEntityNode],
) -> list[TinyEntityNode]:
    search_results: list[TinySearchResult] = await run_in_semaphore(
        *[
            search(
                query=e.name,
                subgraph_ids=[e.subgraph_id],
                clients=clients,
                config=NODE_HYBRID_SEARCH_CROSS_ENCODER,
            )
            for e in extracted_entities
        ]
    )

    duplicite_candidates = {e.uuid: e for result in search_results for e in result.entities}
    return list(duplicite_candidates.values())


def _create_candidates_index(existing_entities: list[TinyEntityNode]) -> EntityCandidateIndex:
    entities_by_uuid: dict[str, TinyEntityNode] = {}
    entities_by_norm_name: dict[str, list[TinyEntityNode]] = defaultdict(list)
    shingles_bu_uuid: dict[str, set[str]] = {}
    lsh_by_uuid: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)

    for candidate in existing_entities:
        norm_exact_name = normalize_string_exact(candidate.name)
        norm_fuzzy_name = normalize_string_for_fuzzy(candidate.name)
        entities_by_norm_name[norm_exact_name].append(candidate)

        name_shingles = shingles(norm_fuzzy_name)
        name_signature = minhash_signature(name_shingles)
        name_lsh = lsh_bands(name_signature)

        entities_by_uuid[candidate.uuid] = candidate
        shingles_bu_uuid[candidate.uuid] = name_shingles
        for band_index, band_hash in enumerate(name_lsh):
            lsh_by_uuid[(band_index, band_hash)].append(candidate.uuid)

    return EntityCandidateIndex(
        entities_by_uuid=entities_by_uuid,
        entities_by_norm_name=entities_by_norm_name,
        shingles_bu_uuid=shingles_bu_uuid,
        lsh_by_uuid=lsh_by_uuid,
    )


def _resolve_with_similarity(
    state: EntityDeduplicationState,
    existing_entity_index: EntityCandidateIndex,
    existing_entities: list[TinyEntityNode],
) -> None:
    for idx, entity in enumerate(existing_entities):
        norm_name = normalize_string_exact(entity.name)
        norm_fuzzy_name = normalize_string_for_fuzzy(entity.name)

        if not has_high_entropy(norm_fuzzy_name):
            state.unresolved_indices.append(idx)
            continue

        # exact match
        exact_matches = existing_entity_index.entities_by_norm_name.get(norm_name, [])
        if len(exact_matches) == 1:
            match = exact_matches[0]
            state.resolved_entities[idx] = match
            state.uuid_map[entity.uuid] = match.uuid
            if match.uuid != entity.uuid:
                state.duplicate_pairs.append((entity, match))
            continue

        if len(exact_matches) > 1:
            state.unresolved_indices.append(idx)
            continue

        # lsh
        existing_shingles = shingles(norm_fuzzy_name)
        existing_signature = minhash_signature(existing_shingles)

        candidates_ids: set[str] = set()
        for band_index, band_name in enumerate(lsh_bands(existing_signature)):
            candidates_ids.update(existing_entity_index.lsh_by_uuid.get((band_index, band_name), []))

        best_candidate: TinyEntityNode | None = None
        best_score: float = 0.0
        for candidate_id in candidates_ids:
            candidate_shingles = existing_entity_index.shingles_bu_uuid.get(candidate_id, set())
            score = jaccard_similarity(existing_shingles, candidate_shingles)
            if score > best_score:
                best_score = score
                best_candidate = existing_entity_index.entities_by_uuid.get(candidate_id)

        if best_candidate and best_score >= _TINY_FUZZY_JACCARD_THRESHOLD:
            state.resolved_entities[idx] = best_candidate
            state.uuid_map[entity.uuid] = best_candidate.uuid
            if best_candidate.uuid != entity.uuid:
                state.duplicate_pairs.append((entity, best_candidate))
            continue

        state.unresolved_indices.append(idx)


async def _resolve_with_llm(
    state: EntityDeduplicationState,
    llm: AbstractLLM,
    existing_entity_index: EntityCandidateIndex,
    existing_entities: list[TinyEntityNode],
    event: TinyEventNode,
) -> None: pass


async def retrieve_events(
    driver: BaseDriver,
    reference_time: datetime,
    last_n: int,
    subgraph_ids: list[str]
) -> list[TinyEventNode]:
    provider = driver.provider
    query = get_last_n_event_nodes(provider)

    if provider == GraphProvider.NEO4J:
        results, _, _ = await driver.execute_query(query, **{
            'reference_time': reference_time,
            'subgraph_ids': subgraph_ids,
            'last_n': last_n,
        })

        return [TinyEventNode.from_record(r) for r in results]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


async def resolve_duplicite_entity_nodes(
    clients: TinyGraphClients,
    extracted_entities: list[TinyEntityNode],
    event_node: TinyEventNode,
) -> list[TinyEntityNode]:
    candidates = await _find_entity_duplicite_candidates(clients, extracted_entities)
    existing_candidates_index = _create_candidates_index(candidates)

    state = EntityDeduplicationState(
        uuid_map={},
        resolved_entities=[None] * len(extracted_entities),
        unresolved_indices=[],
        duplicate_pairs=[],
    )

    _resolve_with_similarity(state, existing_candidates_index, extracted_entities)

    await _resolve_with_llm(state, clients.llm, existing_candidates_index, extracted_entities, event_node)

    # map `uuid_map` a -> a
    for idx, node in enumerate(extracted_entities):
        if state.resolved_entities[idx] is None:
            state.resolved_entities[idx] = node
            state.uuid_map[node.uuid] = node.uuid

    logger.info('extracted new entities %s', extracted_entities)
    logger.info('existing in db entities: %s', candidates)
    logger.info('state: %s', state)

    return []
