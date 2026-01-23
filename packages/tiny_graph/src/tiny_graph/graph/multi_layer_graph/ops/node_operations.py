from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

from pydantic import Field

from tiny_graph.driver.base import BaseDriver
from tiny_graph.graph.multi_layer_graph.core.node import entity_node_batch_embeddings
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEventNode
from tiny_graph.graph.multi_layer_graph.queries.node_queries import (
    get_last_n_event_nodes,
)
from tiny_graph.graph.multi_layer_graph.queries.node_queries import (
    save_cluster_nodes_bulk,
)
from tiny_graph.graph.multi_layer_graph.queries.node_queries import (
    save_entity_nodes_bulk,
)
from tiny_graph.graph.multi_layer_graph.queries.node_queries import save_event_nodes_bulk
from tiny_graph.graph.multi_layer_graph.search.search import search
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchResult
from tiny_graph.graph.multi_layer_graph.search.search_presets import (
    NODE_HYBRID_SEARCH_RRF,
)
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.graph.multi_layer_graph.utils.node_formatter import entity_node_2_prompt
from tiny_graph.graph.multi_layer_graph.utils.node_formatter import event_node_2_prompt
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import (
    _TINY_FUZZY_JACCARD_THRESHOLD,
)
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import has_high_entropy
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import jaccard_similarity
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import lsh_bands
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import minhash_signature
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import (
    normalize_string_exact,
)
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import (
    normalize_string_for_fuzzy,
)
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import shingles
from tiny_graph.types.provider import GraphProvider
from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.runtime.executors import run_in_semaphore
from tinygent.core.types.base import TinyModel
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)


@dataclass
class EntityCandidateIndex:
    existing_entities: list[TinyEntityNode]

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


class EntityDuplicateInfo(TinyModel):
    idx: int = Field(..., description='integer id of the entity')

    duplicate_idx: int = Field(
        ...,
        description='idx of the duplicate entity. If no duplicate entities are found, default to -1.',
    )
    name: str = Field(
        ...,
        description='Name of the entity. Should be the most complete and descriptive name of the entity. Do not include any JSON formatting in the Entity name such as {}.',
    )
    duplicates: list[int] = Field(
        ...,
        description='idx of all entities that are a duplicate of the entity with the above id.',
    )


class EntityResolution(TinyModel):
    entity_resolutions: list[EntityDuplicateInfo] = Field(
        ..., description='List of resolved nodes'
    )


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
                config=NODE_HYBRID_SEARCH_RRF,
            )
            for e in extracted_entities
        ]
    )

    duplicite_candidates = {
        e.uuid: e for result in search_results for e in result.entities
    }
    return list(duplicite_candidates.values())


def _create_candidates_index(
    existing_entities: list[TinyEntityNode],
) -> EntityCandidateIndex:
    entities_by_uuid: dict[str, TinyEntityNode] = {}
    entities_by_norm_name: defaultdict[str, list[TinyEntityNode]] = defaultdict(list)
    shingles_bu_uuid: dict[str, set[str]] = {}
    lsh_by_uuid: defaultdict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)

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
        existing_entities=existing_entities,
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
            candidates_ids.update(
                existing_entity_index.lsh_by_uuid.get((band_index, band_name), [])
            )

        best_candidate: TinyEntityNode | None = None
        best_score: float = 0.0
        for candidate_id in candidates_ids:
            candidate_shingles = existing_entity_index.shingles_bu_uuid.get(
                candidate_id, set()
            )
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


def _resolve_with_llm(
    state: EntityDeduplicationState,
    llm: AbstractLLM,
    existing_entity_index: EntityCandidateIndex,
    extracted_entities: list[TinyEntityNode],
    event: TinyEventNode,
    previous_events: list[TinyEventNode],
    entity_types: dict[str, type[TinyModel]] | None = None,
) -> None:
    from tiny_graph.graph.multi_layer_graph.prompts.nodes import (
        get_llm_resolve_duplicites_prompt_template,
    )

    if not state.unresolved_indices:
        return

    entity_types = entity_types or {}

    unresolved_entities = [extracted_entities[i] for i in state.unresolved_indices]

    unresolved_entities_ctx = [
        {
            'id': i,
            'name': e.name,
            'entity_type': e.labels,
            'entity_type_description': entity_types.get(
                next((item for item in e.labels if item != NodeType.ENTITY.value), ''),
                None,
            ).__doc__
            or 'Default Entity Type',
        }
        for i, e in enumerate(unresolved_entities)
    ]

    existing_entities_ctx = [
        {
            'idx': i,
            'name': candidate.name,
            'entity_types': candidate.labels,
        }
        for i, candidate in enumerate(existing_entity_index.existing_entities)
    ]

    llm_in = get_llm_resolve_duplicites_prompt_template()

    result = llm.generate_structured(
        llm_input=TinyLLMInput(
            messages=[
                TinySystemMessage(content=llm_in.system),
                TinyHumanMessage(
                    content=render_template(
                        llm_in.user,
                        {
                            'previous_events': [
                                event_node_2_prompt(e) for e in previous_events
                            ],
                            'current_event': event_node_2_prompt(event),
                            'extracted_entities': unresolved_entities_ctx,
                            'existing_entities': existing_entities_ctx,
                        },
                    )
                ),
            ]
        ),
        output_schema=EntityResolution,
    )

    entity_resolutions = result.entity_resolutions
    valid_relative_range = range(len(state.unresolved_indices))
    processed_relative_ids: set[int] = set()

    received_ids = {r.idx for r in entity_resolutions}
    expected_ids = set(valid_relative_range)
    missing_ids = expected_ids - received_ids
    extra_ids = received_ids - expected_ids

    logger.debug(
        'Received %d resolutions for %d entities',
        len(entity_resolutions),
        len(state.unresolved_indices),
    )

    if missing_ids:
        logger.warning('LLM did not return resolutions for IDs: %s', sorted(missing_ids))

    if extra_ids:
        logger.warning(
            'LLM returned invalid IDs outside valid range 0-%d: %s (all returned IDs: %s)',
            len(state.unresolved_indices) - 1,
            sorted(extra_ids),
            sorted(received_ids),
        )

    for resolution in entity_resolutions:
        rel_idx = resolution.idx
        dup_idx = resolution.duplicate_idx

        if rel_idx not in valid_relative_range:
            logger.warning(
                'Skipping invalid LLM dedupe id %d (valid range: 0-%d, received %d resolutions)',
                rel_idx,
                len(state.unresolved_indices) - 1,
                len(entity_resolutions),
            )
            continue

        if rel_idx in processed_relative_ids:
            logger.warning('Duplicate LLM dedupe id %s received; ignoring.', rel_idx)
            continue

        processed_relative_ids.add(rel_idx)

        original_index = state.unresolved_indices[rel_idx]
        extracted_node = extracted_entities[original_index]

        resolved_node: TinyEntityNode
        if dup_idx == -1:
            resolved_node = extracted_node
        elif 0 <= dup_idx < len(existing_entity_index.existing_entities):
            resolved_node = existing_entity_index.existing_entities[dup_idx]
        else:
            logger.warning(
                'Invalid duplicate_idx %s for extracted node %s; treating as no duplicate.',
                dup_idx,
                extracted_node.uuid,
            )
            resolved_node = extracted_node

        state.resolved_entities[original_index] = resolved_node
        state.uuid_map[extracted_node.uuid] = resolved_node.uuid
        if resolved_node.uuid != extracted_node.uuid:
            state.duplicate_pairs.append((extracted_node, resolved_node))


async def _extract_entity_attributes(
    llm: AbstractLLM,
    entity_node: TinyEntityNode,
    event_node: TinyEventNode | None = None,
    previous_events: list[TinyEventNode] | None = None,
    entity_type: type[TinyModel] | None = None,
) -> dict[str, Any]:
    if not entity_type:
        return {}

    from tiny_graph.graph.multi_layer_graph.prompts.nodes import (
        get_entity_attributes_extraction_prompt,
    )

    prompt = get_entity_attributes_extraction_prompt()

    response = await llm.agenerate_structured(
        llm_input=TinyLLMInput(
            messages=[
                TinySystemMessage(content=prompt.system),
                TinyHumanMessage(
                    content=render_template(
                        prompt.user,
                        {
                            'entity': entity_node_2_prompt(entity_node),
                            'event_content': (
                                event_node_2_prompt(event_node) if event_node else None
                            ),
                            'previous_events': (
                                [event_node_2_prompt(e) for e in previous_events]
                                if previous_events
                                else None
                            ),
                        },
                    )
                ),
            ]
        ),
        output_schema=entity_type,
    )

    return response.model_dump(mode='json')


async def _extract_entity_summary(
    llm: AbstractLLM,
    entity_node: TinyEntityNode,
    event_node: TinyEventNode | None = None,
    previous_events: list[TinyEventNode] | None = None,
) -> str:
    from tiny_graph.graph.multi_layer_graph.prompts.nodes import (
        get_entity_summary_creation_prompt,
    )

    prompt = get_entity_summary_creation_prompt()

    response = await llm.agenerate_text(
        llm_input=TinyLLMInput(
            messages=[
                TinySystemMessage(content=prompt.system),
                TinyHumanMessage(
                    content=render_template(
                        prompt.user,
                        {
                            'existing_summary': entity_node.summary,
                            'entity': entity_node_2_prompt(entity_node),
                            'event_content': (
                                event_node_2_prompt(event_node) if event_node else None
                            ),
                            'previous_events': (
                                [event_node_2_prompt(e) for e in previous_events]
                                if previous_events
                                else None
                            ),
                        },
                    )
                ),
            ]
        )
    )

    return response.to_string()


async def retrieve_events(
    driver: BaseDriver,
    reference_time: datetime,
    last_n: int,
    subgraph_ids: list[str],
) -> list[TinyEventNode]:
    provider = driver.provider
    query = get_last_n_event_nodes(provider)

    if provider == GraphProvider.NEO4J:
        results, _, _ = await driver.execute_query(
            query,
            **{
                'reference_time': reference_time,
                'subgraph_ids': subgraph_ids,
                'last_n': last_n,
            },
        )

        r_events = [TinyEventNode.from_record(r) for r in results]
        logger.debug(
            'Retrieved (%d) events: %s',
            len(r_events),
            [r.serialized_data for r in r_events],
        )

        return r_events

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


async def resolve_extracted_entity_nodes(
    clients: TinyGraphClients,
    extracted_entities: list[TinyEntityNode],
    event_node: TinyEventNode,
    previous_events: list[TinyEventNode],
    *,
    entity_types: dict[str, type[TinyModel]] | None = None,
) -> tuple[list[TinyEntityNode], dict[str, str]]:
    candidates = await _find_entity_duplicite_candidates(clients, extracted_entities)
    existing_candidates_index = _create_candidates_index(candidates)

    state = EntityDeduplicationState(
        uuid_map={},
        resolved_entities=[None] * len(extracted_entities),
        unresolved_indices=[],
        duplicate_pairs=[],
    )

    _resolve_with_similarity(state, existing_candidates_index, extracted_entities)

    _resolve_with_llm(
        state,
        clients.llm,
        existing_candidates_index,
        extracted_entities,
        event_node,
        previous_events,
        entity_types,
    )

    # map `uuid_map` a -> a
    for idx, node in enumerate(extracted_entities):
        if state.resolved_entities[idx] is None:
            state.resolved_entities[idx] = node
            state.uuid_map[node.uuid] = node.uuid

    return (
        [e for e in state.resolved_entities if e],
        state.uuid_map,
    )


async def extract_attributes_from_node(
    llm: AbstractLLM,
    entity_node: TinyEntityNode,
    event_node: TinyEventNode | None = None,
    previous_events: list[TinyEventNode] | None = None,
    *,
    entity_type: type[TinyModel] | None = None,
) -> TinyEntityNode:
    if entity_type is None or len(entity_type.model_fields) == 0:
        return entity_node

    (attributes, summary) = await run_in_semaphore(
        _extract_entity_attributes(
            llm, entity_node, event_node, previous_events, entity_type
        ),
        _extract_entity_summary(llm, entity_node, event_node, previous_events),
    )

    logger.debug(
        'For entity: %s extracted attributes: %s and summary: %s',
        entity_node.uuid,
        attributes,
        summary,
    )

    entity_node.attributes = attributes
    entity_node.summary = summary
    return entity_node


async def extract_attributes_from_nodes(
    llm: AbstractLLM,
    embedder: AbstractEmbedder,
    entities: list[TinyEntityNode],
    event: TinyEventNode | None = None,
    previous_events: list[TinyEventNode] | None = None,
    *,
    entity_types: dict[str, type[TinyModel]] | None = None,
) -> list[TinyEntityNode]:
    updated_entities = await run_in_semaphore(
        *[
            extract_attributes_from_node(
                llm,
                entity,
                event,
                previous_events,
                entity_type=(
                    entity_types.get(
                        next(
                            (
                                item
                                for item in entity.labels
                                if item != NodeType.ENTITY.value
                            ),
                            '',
                        )
                    )
                    if entity_types is not None
                    else None
                ),
            )
            for entity in entities
        ]
    )

    embedded_entities = await entity_node_batch_embeddings(embedder, updated_entities)
    return embedded_entities


async def bulk_save_entities(
    driver: BaseDriver, entities: list[TinyEntityNode]
) -> list[str]:
    payload = [
        {
            'uuid': e.uuid,
            'name': e.name,
            'subgraph_id': e.subgraph_id,
            'created_at': e.created_at,
            'summary': e.summary,
            'labels': e.labels,
            'name_embedding': e.name_embedding,
        }
        for e in entities
    ]

    results, _, _ = await driver.execute_query(
        query=save_entity_nodes_bulk(driver.provider),
        entities=payload,
    )

    saved_uuids = results[0]['uuids'] if results else []
    logger.debug('Saved %d entity nodes', len(saved_uuids))

    return saved_uuids


async def bulk_save_events(driver: BaseDriver, events: list[TinyEventNode]) -> list[str]:
    payload = [
        {
            'uuid': e.uuid,
            'name': e.name,
            'description': e.description,
            'subgraph_id': e.subgraph_id,
            'created_at': e.created_at,
            'valid_at': e.valid_at,
            'data': e.serialized_data,
            'data_type': e.data_type.value,
        }
        for e in events
    ]

    results, _, _ = await driver.execute_query(
        query=save_event_nodes_bulk(driver.provider),
        events=payload,
    )

    saved_uuids = results[0]['uuids'] if results else []
    logger.debug('Saved %d event nodes', len(saved_uuids))

    return saved_uuids


async def bulk_save_clusters(
    driver: BaseDriver, clusters: list[TinyClusterNode]
) -> list[str]:
    payload = [
        {
            'uuid': c.uuid,
            'name': c.name,
            'subgraph_id': c.subgraph_id,
            'created_at': c.created_at,
            'summary': c.summary,
            'name_embedding': c.name_embedding,
        }
        for c in clusters
    ]

    results, _, _ = await driver.execute_query(
        query=save_cluster_nodes_bulk(driver.provider),
        clusters=payload,
    )

    saved_uuids = results[0]['uuids'] if results else []
    logger.debug('Saved %d cluster nodes', len(saved_uuids))

    return saved_uuids
