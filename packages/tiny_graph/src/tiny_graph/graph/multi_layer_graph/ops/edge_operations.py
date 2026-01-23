from datetime import datetime
import json
import logging
import os
from typing import Any

from pydantic import Field

from tiny_graph.driver.base import BaseDriver
from tiny_graph.graph.multi_layer_graph.core.edge import entity_edge_batch_embeddings
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.edges import TinyClusterEdge
from tiny_graph.graph.multi_layer_graph.edges import TinyEntityEdge
from tiny_graph.graph.multi_layer_graph.edges import TinyEventEdge
from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEventNode
from tiny_graph.graph.multi_layer_graph.search.search import search
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchFilters
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchResult
from tiny_graph.graph.multi_layer_graph.search.search_presets import (
    EDGE_HYBRID_SEARCH_RRF,
)
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.graph.multi_layer_graph.utils.text_similarity import (
    normalize_string_exact,
)
from tiny_graph.helper import ensure_utc
from tiny_graph.helper import get_current_timestamp
from tiny_graph.helper import parse_timestamp
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.runtime.executors import run_in_semaphore
from tinygent.core.types.base import TinyModel
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

DEFAULT_EDGE_NAME = os.getenv('TINY_DEFAULT_EDGE_NAME', 'RELATES_TO')
MAX_REFLEXION_ITERATIONS = int(os.getenv('TINY_GRAPH_MAX_REFLEXION_ITERATIONS', 0))


class ClusterEdge(TinyModel):
    source_cluster_id: int = Field(
        ..., description='The id of the source entity from the CLUSTERS list'
    )
    target_entity_id: int = Field(
        ..., description='The id of the target entity from the ENTITIES list'
    )


class ExtractedClusterEdges(TinyModel):
    edges: list[ClusterEdge]


class EntityEdge(TinyModel):
    relation_type: str = Field(..., description='FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE')
    source_entity_id: int = Field(
        ..., description='The id of the source entity from the ENTITIES list'
    )
    target_entity_id: int = Field(
        ..., description='The id of the target entity from the ENTITIES list'
    )
    fact: str = Field(
        ...,
        description='A natural language description of the relationship between the entities, paraphrased from the source text',
    )
    valid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )


class ExtractedEntityEdges(TinyModel):
    edges: list[EntityEdge]


class MissingFacts(TinyModel):
    missing_facts: list[str] = Field(..., description="facts that weren't extracted")


class EdgeDuplicate(TinyModel):
    duplicate_facts: list[int] = Field(
        ...,
        description='List of idx values of any duplicate facts. If no duplicate facts are found, default to empty list.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of idx values of facts that should be invalidated. If no facts should be invalidated, the list should be empty.',
    )
    fact_type: str = Field(..., description='One of the provided fact types or DEFAULT')


async def _extract_entity_edges(
    llm: AbstractLLM,
    entities: list[TinyEntityNode],
    event_node: TinyEventNode,
    previous_events: list[TinyEventNode],
    edge_types: dict[str, type[TinyModel]] | None,
    edge_type_map: dict[tuple[str, str], list[str]],
    subgraph_id: str,
) -> list[TinyEntityEdge]:
    edge_type_signature_map: dict[str, tuple[str, str]] = {
        edge_type: signature
        for signature, edge_types in edge_type_map.items()
        for edge_type in edge_types
    }

    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_signature': edge_type_signature_map.get(
                    type_name, (NodeType.ENTITY.value, NodeType.ENTITY.value)
                ),
                'fact_type_description': type_model.doc_cls(),
            }
            for type_name, type_model in edge_types.items()
        ]
        if edge_types is not None
        else []
    )

    # Prepare context for LLM
    context: dict[str, Any] = {
        'event_content': event_node.serialized_data,
        'entities': [
            {'id': idx, 'name': node.name, 'entity_types': node.labels}
            for idx, node in enumerate(entities)
        ],
        'previous_events': [ep.serialized_data for ep in previous_events],
        'reference_time': event_node.valid_at,
        'edge_types': edge_types_context,
        'custom_prompt': '',
        'current_date': parse_timestamp(get_current_timestamp()),
    }

    facts_missed = True
    reflexion_iterations = 0
    edges_data: list[EntityEdge] | None = None

    from tiny_graph.graph.multi_layer_graph.prompts.edges import (
        get_entity_edge_extraction_prompt,
    )

    edge_prompt = get_entity_edge_extraction_prompt()

    # extract edges & check all missing facts using reflexion
    while facts_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
        result = llm.generate_structured(
            llm_input=TinyLLMInput(
                messages=[
                    TinySystemMessage(
                        content=edge_prompt.system,
                    ),
                    TinyHumanMessage(
                        content=render_template(
                            edge_prompt.user,
                            context,
                        )
                    ),
                ]
            ),
            output_schema=ExtractedEntityEdges,
        )

        edges_data = result.edges
        context['extracted_facts'] = [edge_data.fact for edge_data in edges_data]

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            from tiny_graph.graph.multi_layer_graph.prompts.edges import (
                get_entity_edge_extract_reflextion_prompt,
            )

            reflex_prompt = get_entity_edge_extract_reflextion_prompt()
            reflexion_result = llm.generate_structured(
                llm_input=TinyLLMInput(
                    messages=[
                        TinySystemMessage(content=reflex_prompt.system),
                        TinyHumanMessage(
                            content=render_template(
                                reflex_prompt.user,
                                context,
                            )
                        ),
                    ]
                ),
                output_schema=MissingFacts,
            )

            if reflexion_result.missing_facts:
                context['custom_prompt'] = (
                    f'The following facts were missed in a previous extraction: {"\n".join(reflexion_result.missing_facts)}'
                )

            facts_missed = bool(reflexion_result.missing_facts)

    if not edges_data:
        return []

    # convert edges data `list[EntityEdge]` -> `list[TinyEntityEdge]`
    edges = []
    for edge_data in edges_data:
        # Validate Edge Date information
        valid_at = edge_data.valid_at
        invalid_at = edge_data.invalid_at
        valid_at_datetime = None
        invalid_at_datetime = None

        # Filter out empty edges
        if not edge_data.fact.strip():
            continue

        source_node_idx = edge_data.source_entity_id
        target_node_idx = edge_data.target_entity_id

        if len(entities) == 0:
            logger.warning('No entities provided for edge extraction')
            continue

        if not (
            0 <= source_node_idx < len(entities) and 0 <= target_node_idx < len(entities)
        ):
            logger.warning(
                f'Invalid entity IDs in edge extraction for {edge_data.relation_type}. '
                f'source_entity_id: {source_node_idx}, target_entity_id: {target_node_idx}, '
                f'but only {len(entities)} entities available (valid range: 0-{len(entities) - 1})'
            )
            continue

        source_node_uuid = entities[source_node_idx].uuid
        target_node_uuid = entities[target_node_idx].uuid

        if valid_at:
            try:
                valid_at_datetime = ensure_utc(
                    datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(
                    f'WARNING: Error parsing valid_at date: {e}. Input: {valid_at}'
                )

        if invalid_at:
            try:
                invalid_at_datetime = ensure_utc(
                    datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(
                    f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}'
                )

        edge = TinyEntityEdge(
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            name=edge_data.relation_type,
            subgraph_id=subgraph_id,
            fact=edge_data.fact,
            events=[event_node.uuid],
            created_at=get_current_timestamp(),
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
        )
        edges.append(edge)
        logger.debug(
            f'Created new edge: {edge.name} from (UUID: {edge.source_node_uuid}) to (UUID: {edge.target_node_uuid})'
        )

    logger.debug(
        'EXTRACTED edges: %s',
        [(e.name, e.source_node_uuid, e.target_node_uuid, e.fact) for e in edges],
    )
    return edges


async def _extract_cluster_edges(
    llm: AbstractLLM,
    event_node: TinyEventNode,
    entities: list[TinyEntityNode],
    extracted_clusters: list[TinyClusterNode],
    existing_clusters: list[TinyClusterNode],
    subgraph_id: str,
) -> list[TinyClusterEdge]:
    all_clusters = extracted_clusters + existing_clusters

    context: dict[str, Any] = {
        'event_content': event_node.serialized_data,
        'entities': [
            {'id': idx, 'name': node.name, 'entity_types': node.labels}
            for idx, node in enumerate(entities)
        ],
        'clusters': [
            {'id': idx, 'name': cluster.name, 'summary': cluster.summary}
            for idx, cluster in enumerate(all_clusters)
        ],
        'custom_prompt': '',
    }

    facts_missed = True
    reflexion_iterations = 0
    valid_extracted_edges: list[ClusterEdge] | None = None

    from tiny_graph.graph.multi_layer_graph.prompts.edges import (
        get_cluster_edge_extraction_prompt,
    )

    edge_prompt = get_cluster_edge_extraction_prompt()

    # extract edges & check all missing facts using reflexion
    while facts_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
        result = llm.generate_structured(
            llm_input=TinyLLMInput(
                messages=[
                    TinySystemMessage(content=edge_prompt.system),
                    TinyHumanMessage(
                        content=render_template(
                            edge_prompt.user,
                            context,
                        )
                    ),
                ]
            ),
            output_schema=ExtractedClusterEdges,
        )

        edges_data = result.edges
        context['extracted_edges'] = []

        # validate results
        valid_extracted_edges = []

        for extracted_edge in edges_data:
            source_cluster_idx = extracted_edge.source_cluster_id
            target_entity_idx = extracted_edge.target_entity_id

            if not (
                0 <= source_cluster_idx < len(all_clusters)
                and 0 <= target_entity_idx < len(entities)
            ):
                logger.warning(
                    f'Invalid cluster/entity IDs in cluster edge extraction. '
                    f'source_cluster_id: {source_cluster_idx} (valid range: 0-{len(all_clusters) - 1}), '
                    f'target_entity_id: {target_entity_idx} (valid range: 0-{len(entities) - 1})'
                )
                continue

            valid_extracted_edges.append(extracted_edge)
            context['extracted_edges'].append(
                {
                    'cluster_name': all_clusters[source_cluster_idx].name,
                    'entity_name': entities[target_entity_idx].name,
                }
            )

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            from tiny_graph.graph.multi_layer_graph.prompts.edges import (
                get_cluster_edge_extract_reflextion_prompt,
            )

            reflex_prompt = get_cluster_edge_extract_reflextion_prompt()
            reflexion_result = llm.generate_structured(
                llm_input=TinyLLMInput(
                    messages=[
                        TinySystemMessage(content=reflex_prompt.system),
                        TinyHumanMessage(
                            content=render_template(
                                reflex_prompt.user,
                                context,
                            )
                        ),
                    ]
                ),
                output_schema=MissingFacts,
            )

            if reflexion_result.missing_facts:
                context['custom_prompt'] = (
                    f'The following facts were missed in a previous extraction: {"\n".join(reflexion_result.missing_facts)}'
                )

            facts_missed = bool(reflexion_result.missing_facts)

    if not valid_extracted_edges:
        return []

    final_edges: list[TinyClusterEdge] = []

    for edge in valid_extracted_edges:
        source_cluster_idx = edge.source_cluster_id
        target_entity_idx = edge.target_entity_id

        cluster_edge = TinyClusterEdge(
            subgraph_id=subgraph_id,
            source_node_uuid=all_clusters[source_cluster_idx].uuid,
            target_node_uuid=entities[target_entity_idx].uuid,
        )
        final_edges.append(cluster_edge)
        logger.debug(
            f'Created new cluster edge: from cluster (UUID: {cluster_edge.source_node_uuid}) to entity (UUID: {cluster_edge.target_node_uuid})'
        )

    logger.debug(
        'EXTRACTED cluster edges: %s',
        [(e.source_node_uuid, e.target_node_uuid) for e in final_edges],
    )
    return final_edges


async def extract_edges(
    llm: AbstractLLM,
    entities: list[TinyEntityNode],
    event_node: TinyEventNode,
    previous_events: list[TinyEventNode],
    edge_types: dict[str, type[TinyModel]] | None,
    edge_type_map: dict[tuple[str, str], list[str]],
    extracted_clusters: list[TinyClusterNode],
    existing_clusters: list[TinyClusterNode],
    subgraph_id: str,
) -> tuple[list[TinyEntityEdge], list[TinyClusterEdge]]:
    extracted_entity_edges, extracted_cluster_edges = await run_in_semaphore(
        _extract_entity_edges(
            llm,
            entities,
            event_node,
            previous_events,
            edge_types,
            edge_type_map,
            subgraph_id,
        ),
        _extract_cluster_edges(
            llm,
            event_node,
            entities,
            extracted_clusters,
            existing_clusters,
            subgraph_id,
        ),
    )

    return extracted_entity_edges, extracted_cluster_edges


def resolve_edge_pointers(
    edges: list[TinyEntityEdge], entity_uuid_map: dict[str, str]
) -> list[TinyEntityEdge]:
    for e in edges:
        e.source_node_uuid = entity_uuid_map.get(e.source_node_uuid, e.source_node_uuid)
        e.target_node_uuid = entity_uuid_map.get(e.target_node_uuid, e.target_node_uuid)

    return edges


def resolve_edge_contradictions(
    resolved_edge: TinyEntityEdge, invalidation_candidates: list[TinyEntityEdge]
) -> list[TinyEntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[TinyEntityEdge] = []
    for edge in invalidation_candidates:
        logger.debug(
            'CHECK contradiction | existing=%s (%s→%s) | new_valid_at=%s | old_valid_at=%s',
            edge.fact,
            edge.source_node_uuid,
            edge.target_node_uuid,
            resolved_edge.valid_at,
            edge.valid_at,
        )

        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        edge_invalid_at_utc = ensure_utc(edge.invalid_at)
        resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
        edge_valid_at_utc = ensure_utc(edge.valid_at)
        resolved_edge_invalid_at_utc = ensure_utc(resolved_edge.invalid_at)

        if (
            edge_invalid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_invalid_at_utc <= resolved_edge_valid_at_utc
        ) or (
            edge_valid_at_utc is not None
            and resolved_edge_invalid_at_utc is not None
            and resolved_edge_invalid_at_utc <= edge_valid_at_utc
        ):
            continue
        # New edge invalidates edge
        elif (
            edge_valid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_valid_at_utc <= resolved_edge_valid_at_utc
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = (
                edge.expired_at
                if edge.expired_at is not None
                else get_current_timestamp()
            )
            invalidated_edges.append(edge)

    return invalidated_edges


async def resolve_extracted_edge(
    llm: AbstractLLM,
    extracted_edge: TinyEntityEdge,
    related_edges: list[TinyEntityEdge],
    existing_edges: list[TinyEntityEdge],
    event: TinyEventNode,
    edge_type_candidates: dict[str, type[TinyModel]] | None = None,
    custom_edge_type_names: set[str] | None = None,
) -> tuple[TinyEntityEdge, list[TinyEntityEdge], list[TinyEntityEdge]]:
    if not (related_edges and existing_edges):
        return extracted_edge, [], []

    # fast check
    normalized_fact = normalize_string_exact(extracted_edge.fact)
    for edge in related_edges:
        if (
            edge.source_node_uuid == extracted_edge.source_node_uuid
            and edge.target_node_uuid == extracted_edge.target_node_uuid
            and normalize_string_exact(edge.fact) == normalized_fact
        ):
            resolved = edge
            if event is not None and event.uuid not in resolved.events:
                resolved.events.append(event.uuid)
            return resolved, [], []

    # Prepare context for LLM
    related_edges_context = [
        {'idx': i, 'fact': edge.fact} for i, edge in enumerate(related_edges)
    ]

    invalidation_edge_candidates_context = [
        {'idx': i, 'fact': existing_edge.fact}
        for i, existing_edge in enumerate(existing_edges)
    ]

    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_description': type_model.doc_cls(),
            }
            for type_name, type_model in edge_type_candidates.items()
        ]
        if edge_type_candidates is not None
        else []
    )

    context = {
        'existing_edges': related_edges_context,
        'new_edge': extracted_edge.fact,
        'edge_invalidation_candidates': invalidation_edge_candidates_context,
        'edge_types': edge_types_context,
    }

    from tiny_graph.graph.multi_layer_graph.prompts.edges import get_resolve_edge_prompt

    resolve_dup_prompt = get_resolve_edge_prompt()
    duplicate_result = llm.generate_structured(
        llm_input=TinyLLMInput(
            messages=[
                TinySystemMessage(content=resolve_dup_prompt.system),
                TinyHumanMessage(
                    content=render_template(
                        resolve_dup_prompt.user,
                        context,
                    )
                ),
            ]
        ),
        output_schema=EdgeDuplicate,
    )

    logger.debug(
        'EDGE RESOLVE decision | new_fact=%r | duplicates=%s | contradicted=%s | fact_type=%s',
        extracted_edge.fact,
        duplicate_result.duplicate_facts,
        duplicate_result.contradicted_facts,
        duplicate_result.fact_type,
    )

    duplicate_facts = duplicate_result.duplicate_facts

    # Validate duplicate_facts are in valid range for EXISTING FACTS
    duplicate_fact_ids: list[int] = [
        i for i in duplicate_facts if 0 <= i < len(related_edges)
    ]

    resolved_edge = extracted_edge
    for duplicate_fact_id in duplicate_fact_ids:
        resolved_edge = related_edges[duplicate_fact_id]
        break

    if duplicate_fact_ids and event is not None:
        resolved_edge.events.append(event.uuid)

    contradicted_facts: list[int] = duplicate_result.contradicted_facts

    # Validate contradicted_facts are in valid range for INVALIDATION CANDIDATES
    invalidation_candidates: list[TinyEntityEdge] = [
        existing_edges[i] for i in contradicted_facts if 0 <= i < len(existing_edges)
    ]

    fact_type: str = duplicate_result.fact_type
    candidate_type_names = set(edge_type_candidates or {})
    custom_type_names = custom_edge_type_names or set()

    is_default_type = fact_type.upper() == 'DEFAULT'
    is_custom_type = fact_type in custom_type_names
    is_allowed_custom_type = fact_type in candidate_type_names

    if is_allowed_custom_type:
        # The LLM selected a custom type that is allowed for the node pair.
        # Adopt the custom type and, if needed, extract its structured attributes.
        resolved_edge.name = fact_type

        edge_attributes_context = {
            'event_content': event.serialized_data,
            'reference_time': event.valid_at,
            'fact': resolved_edge.fact,
        }

        edge_model = (
            edge_type_candidates.get(fact_type) if edge_type_candidates else None
        )
        if edge_model is not None and len(edge_model.model_fields) != 0:
            from tiny_graph.graph.multi_layer_graph.prompts.edges import (
                get_extract_edge_attributes,
            )

            edge_attrs_promp = get_extract_edge_attributes()

            edge_attributes_response = llm.generate_structured(
                llm_input=TinyLLMInput(
                    messages=[
                        TinySystemMessage(content=edge_attrs_promp.system),
                        TinyHumanMessage(
                            content=render_template(
                                edge_attrs_promp.user,
                                edge_attributes_context,
                            )
                        ),
                    ]
                ),
                output_schema=edge_model,
            )

            resolved_edge.attributes = edge_attributes_response.model_dump(mode='json')
    elif not is_default_type and is_custom_type:
        # The LLM picked a custom type that is not allowed for this signature.
        # Reset to the default label and drop any structured attributes.
        resolved_edge.name = DEFAULT_EDGE_NAME
        resolved_edge.attributes = {}
    elif not is_default_type:
        # Non-custom labels are allowed to pass through so long as the LLM does
        # not return the sentinel DEFAULT value.
        resolved_edge.name = fact_type
        resolved_edge.attributes = {}

    now = get_current_timestamp()

    if resolved_edge.invalid_at and not resolved_edge.expired_at:
        resolved_edge.expired_at = now

    # Determine if the new_edge needs to be expired
    if resolved_edge.expired_at is None:
        invalidation_candidates.sort(
            key=lambda c: (c.valid_at is None, ensure_utc(c.valid_at))
        )
        for candidate in invalidation_candidates:
            candidate_valid_at_utc = ensure_utc(candidate.valid_at)
            resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
            if (
                candidate_valid_at_utc is not None
                and resolved_edge_valid_at_utc is not None
                and candidate_valid_at_utc > resolved_edge_valid_at_utc
            ):
                # Expire new edge since we have information about more recent events
                resolved_edge.invalid_at = candidate.valid_at
                resolved_edge.expired_at = now
                break

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[TinyEntityEdge] = resolve_edge_contradictions(
        resolved_edge, invalidation_candidates
    )
    duplicate_edges: list[TinyEntityEdge] = [
        related_edges[idx] for idx in duplicate_fact_ids
    ]

    return resolved_edge, invalidated_edges, duplicate_edges


async def resolve_extracted_entity_edges(
    clients: TinyGraphClients,
    extracted_edges: list[TinyEntityEdge],
    event: TinyEventNode,
    entities: list[TinyEntityNode],
    entity_uuid_map: dict[str, str],
    edge_types: dict[str, type[TinyModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
) -> tuple[list[TinyEntityEdge], list[TinyEntityEdge]]:
    extracted_edges = resolve_edge_pointers(extracted_edges, entity_uuid_map)

    # exact fast deduplication
    deduplicated_edges: list[TinyEntityEdge] = []
    seen: set[tuple[str, str, str]] = set()

    for e in extracted_edges:
        signature = (
            e.source_node_uuid,
            e.target_node_uuid,
            normalize_string_exact(e.fact),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduplicated_edges.append(e)

    extracted_edges = deduplicated_edges

    # create embeddings for all edges
    await entity_edge_batch_embeddings(clients.embedder, extracted_edges)

    # answers question: What edges are in db for same entity node pairs as we just extracted?
    # used for deduplication
    all_already_existing_edges: list[list[TinyEntityEdge]] = await run_in_semaphore(
        *[
            TinyEntityEdge.find_by_targets(
                clients.driver, edge.source_node_uuid, edge.target_node_uuid
            )
            for edge in extracted_edges
        ]
    )

    for extracted_edge, existing in zip(
        extracted_edges, all_already_existing_edges, strict=True
    ):
        logger.debug(
            'DB edges for (%s -> %s): %s',
            extracted_edge.source_node_uuid,
            extracted_edge.target_node_uuid,
            [(e.uuid, e.fact) for e in existing],
        )

    # answers question: What similar edges for those i just extracted we already have in db?
    similar_edges_search_result: list[TinySearchResult] = await run_in_semaphore(
        *[
            search(
                query=extracted_edge.fact,
                clients=clients,
                subgraph_ids=[extracted_edge.subgraph_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                filters=TinySearchFilters(edge_uuids=[e.uuid for e in valid_edges]),
            )
            for extracted_edge, valid_edges in zip(
                extracted_edges, all_already_existing_edges, strict=True
            )
        ]
    )

    semantic_similar_edges: list[list[TinyEntityEdge]] = [
        r.edges for r in similar_edges_search_result
    ]

    # getting semantically similar/related edges as potential candidates for same/contradict older edges for invalidation
    edge_invalidation_candidates_search_result: list[
        TinySearchResult
    ] = await run_in_semaphore(
        *[
            search(
                query=extracted_edge.fact,
                clients=clients,
                subgraph_ids=[extracted_edge.subgraph_id],
                config=EDGE_HYBRID_SEARCH_RRF,
            )
            for extracted_edge in extracted_edges
        ]
    )

    edge_invalidation_candidates: list[list[TinyEntityEdge]] = [
        r.edges for r in edge_invalidation_candidates_search_result
    ]

    for extracted_edge, candidates in zip(
        extracted_edges, edge_invalidation_candidates, strict=True
    ):
        logger.debug(
            'INVALIDATION candidates for new fact [%s]: %s',
            extracted_edge.fact,
            [(e.uuid, e.fact) for e in candidates],
        )

    # Build entity hash table
    uuid_entity_map: dict[str, TinyEntityNode] = {
        entity.uuid: entity for entity in entities
    }

    # Determine which edge types are relevant for each edge.
    # `edge_types_lst` stores the subset of custom edge definitions whose
    # node signature matches each extracted edge. Anything outside this subset
    # should only stay on the edge if it is a non-custom (LLM generated) label.
    edge_types_lst: list[dict[str, type[TinyModel]]] = []
    custom_type_names = set(edge_types or {})

    for extracted_edge in extracted_edges:
        source_node = uuid_entity_map.get(extracted_edge.source_node_uuid)
        target_node = uuid_entity_map.get(extracted_edge.target_node_uuid)

        source_node_labels: list[str] = (
            source_node.labels + [NodeType.ENTITY.value]
            if source_node is not None
            else [NodeType.ENTITY.value]
        )
        target_node_labels: list[str] = (
            target_node.labels + [NodeType.ENTITY.value]
            if target_node is not None
            else [NodeType.ENTITY.value]
        )
        label_tuples: list[tuple[str, str]] = [
            (source_label, target_label)
            for source_label in source_node_labels
            for target_label in target_node_labels
        ]

        extracted_edge_types = {}
        for label_tuple in label_tuples:
            type_names = edge_type_map.get(label_tuple, [])
            for type_name in type_names:
                type_model = edge_types.get(type_name)
                if type_model is None:
                    continue

                extracted_edge_types[type_name] = type_model

        edge_types_lst.append(extracted_edge_types)

    # resolve edges with related edges in the graph and find invalidation candidates
    results: list[tuple[TinyEntityEdge, list[TinyEntityEdge], list[TinyEntityEdge]]] = (
        list(
            await run_in_semaphore(
                *[
                    resolve_extracted_edge(
                        clients.llm,
                        extracted_edge,
                        related_edges,
                        existing_edges,
                        event,
                        extracted_edge_types,
                        custom_type_names,
                    )
                    for extracted_edge, related_edges, existing_edges, extracted_edge_types in zip(
                        extracted_edges,
                        semantic_similar_edges,
                        edge_invalidation_candidates,
                        edge_types_lst,
                        strict=True,
                    )
                ]
            )
        )
    )

    resolved_edges: list[TinyEntityEdge] = []
    seen_resolved: set[str] = set()

    invalidated_edges: list[TinyEntityEdge] = []
    seen_invalidated: set[str] = set()

    for result in results:
        resolved_edge = result[0]
        invalidated_edge_chunk = result[1]

        if resolved_edge.uuid not in seen_resolved:
            seen_resolved.add(resolved_edge.uuid)
            resolved_edges.append(resolved_edge)

        for e in invalidated_edge_chunk:
            if e.uuid not in seen_invalidated:
                seen_invalidated.add(e.uuid)
                invalidated_edges.append(e)

    logger.debug(f'Resolved edges: {[(e.name, e.uuid) for e in resolved_edges]}')

    await run_in_semaphore(
        entity_edge_batch_embeddings(clients.embedder, resolved_edges),
        entity_edge_batch_embeddings(clients.embedder, invalidated_edges),
    )
    return resolved_edges, invalidated_edges


async def bulk_save_entity_edges(
    driver: BaseDriver, edges: list[TinyEntityEdge]
) -> list[str]:
    from tiny_graph.graph.multi_layer_graph.queries.edge_queries import (
        save_entity_edges_bulk,
    )

    payload = [
        {
            'uuid': e.uuid,
            'subgraph_id': e.subgraph_id,
            'source_node_uuid': e.source_node_uuid,
            'target_node_uuid': e.target_node_uuid,
            'created_at': e.created_at,
            'name': e.name,
            'fact': e.fact,
            'fact_embedding': e.fact_embedding,
            'events': e.events,
            'expired_at': e.expired_at,
            'valid_at': e.valid_at,
            'invalid_at': e.invalid_at,
            'attributes': json.dumps(e.attributes),
        }
        for e in edges
    ]

    results, _, _ = await driver.execute_query(
        query=save_entity_edges_bulk(driver.provider),
        edges=payload,
    )

    saved_uuids = results[0]['uuids'] if results else []
    logger.debug('Saved %d entity edges', len(saved_uuids))

    return saved_uuids


async def bulk_save_cluster_edges(
    driver: BaseDriver, edges: list[TinyClusterEdge]
) -> list[str]:
    from tiny_graph.graph.multi_layer_graph.queries.edge_queries import (
        save_cluster_edges_bulk,
    )

    payload = [
        {
            'uuid': e.uuid,
            'subgraph_id': e.subgraph_id,
            'created_at': e.created_at,
            'cluster_node_uuid': e.source_node_uuid,
            'entity_node_uuid': e.target_node_uuid,
        }
        for e in edges
    ]

    results, _, _ = await driver.execute_query(
        query=save_cluster_edges_bulk(driver.provider),
        edges=payload,
    )

    saved_uuids = results[0]['uuids'] if results else []
    logger.debug('Saved %d cluster edges', len(saved_uuids))

    return saved_uuids


async def bulk_save_event_edges(
    driver: BaseDriver, edges: list[TinyEventEdge]
) -> list[str]:
    from tiny_graph.graph.multi_layer_graph.queries.edge_queries import (
        save_event_edges_bulk,
    )

    payload = [
        {
            'uuid': e.uuid,
            'subgraph_id': e.subgraph_id,
            'created_at': e.created_at,
            'event_node_uuid': e.source_node_uuid,
            'entity_node_uuid': e.target_node_uuid,
        }
        for e in edges
    ]

    results, _, _ = await driver.execute_query(
        query=save_event_edges_bulk(driver.provider),
        edges=payload,
    )

    saved_uuids = results[0]['uuids'] if results else []
    logger.debug('Saved %d event edges', len(saved_uuids))

    return saved_uuids
