from collections import defaultdict
import logging

from pydantic import Field

from tiny_graph.graph.multi_layer_graph.core.node import cluster_node_batch_embeddings
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.search.search import search
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchFilters
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinySearchResult
from tiny_graph.graph.multi_layer_graph.search.search_presets import (
    CLUSTER_HYBRID_SEARCH_RRF,
)
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.runtime.executors import run_in_semaphore
from tinygent.core.types.base import TinyModel
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)


class NewClusterProposal(TinyModel):
    name: str = Field(..., description='Name of newly proposed cluster')

    summary: str = Field(
        ...,
        description='Should be the most complete and descriptive name of the entity. Do not include any JSON formatting in the Entity name such as {}.',
    )


class NewClusterProposals(TinyModel):
    proposals: list[NewClusterProposal] = Field(
        ..., description='List of newly proposed clusters.'
    )


async def resolve_and_extract_clusters(
    clients: TinyGraphClients,
    entities: list[TinyEntityNode],
) -> tuple[list[TinyClusterNode], list[TinyClusterNode]]:
    # get all existing / valid clusters for current entities
    all_already_existing_clusters: list[list[TinyClusterNode]] = await run_in_semaphore(
        *[
            TinyClusterNode.find_by_entity(clients.driver, entity.uuid)
            for entity in entities
        ]
    )

    existing_clusters = {
        c.uuid: c for clusters in all_already_existing_clusters for c in clusters
    }

    cluster_entities_map: defaultdict[str, list[TinyEntityNode]] = defaultdict(list)

    for entity, clusters in zip(entities, all_already_existing_clusters, strict=True):
        for cluster in clusters:
            cluster_entities_map[cluster.uuid].append(entity)

        logger.debug(
            'Existing clusters for entity (%s): %s',
            entity.uuid,
            [c.uuid for c in clusters],
        )

    # answers: What similar clusters we have from all existing clusters to current entity
    similar_clusters_search_results: list[TinySearchResult] = await run_in_semaphore(
        *[
            search(
                query=entity.name,
                clients=clients,
                subgraph_ids=[entity.subgraph_id],
                config=CLUSTER_HYBRID_SEARCH_RRF,
                filters=TinySearchFilters(
                    cluster_uuids=[c.uuid for c in all_already_existing_clusters[i]]
                ),
            )
            for i, entity in enumerate(entities)
        ]
    )

    semantic_similar_clusters: list[list[TinyClusterNode]] = [
        r.clusters for r in similar_clusters_search_results
    ]

    semantic_clusters = {
        c.uuid: c for clusters in semantic_similar_clusters for c in clusters
    }

    logger.debug(
        'Semantic similar clusters for %s', [c for c in semantic_clusters.keys()]
    )

    cluster_entity_context = {
        'existing_clusters': [
            {
                'name': c.name,
                'summary': c.summary,
                'entities': [
                    {
                        'entity_name': e.name,
                        'entity_summary': e.summary,
                    }
                    for e in cluster_entities_map.get(c.uuid, [])
                ],
            }
            for c in existing_clusters.values()
        ],
        'candidate_clusters': [
            {
                'name': c.name,
                'summary': c.summary,
            }
            for c in semantic_clusters.values()
            if c.uuid not in existing_clusters
        ],
        'entities': [
            {
                'entity_name': e.name,
                'entity_summary': e.summary,
            }
            for e in entities
        ],
    }

    from tiny_graph.graph.multi_layer_graph.prompts.clusters import (
        get_new_clusters_proposal_prompt,
    )

    new_proposals_prompt = get_new_clusters_proposal_prompt()

    new_cluster_proposals = clients.llm.generate_structured(
        llm_input=TinyLLMInput(
            messages=[
                TinySystemMessage(content=new_proposals_prompt.system),
                TinyHumanMessage(
                    content=render_template(
                        new_proposals_prompt.user,
                        cluster_entity_context,
                    )
                ),
            ]
        ),
        output_schema=NewClusterProposals,
    )

    logger.debug('new clusters: %s', new_cluster_proposals.proposals)

    extracted_clusters = [
        TinyClusterNode(
            name=p.name,
            subgraph_id=next((e.subgraph_id for e in entities if e.subgraph_id), ''),
            summary=p.summary,
        )
        for p in new_cluster_proposals.proposals
    ]

    extracted_clusters = await cluster_node_batch_embeddings(
        clients.embedder, extracted_clusters
    )

    return extracted_clusters, list(semantic_clusters.values())
