from datetime import datetime
import json
import logging
import time

from tiny_graph.driver.base import BaseDriver
from tiny_graph.graph.base import BaseGraph
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.datamodels.extract_nodes import ExtractedEntities
from tiny_graph.graph.multi_layer_graph.datamodels.extract_nodes import ExtractedEntity
from tiny_graph.graph.multi_layer_graph.datamodels.extract_nodes import MissedEntities
from tiny_graph.graph.multi_layer_graph.edges import TinyClusterEdge
from tiny_graph.graph.multi_layer_graph.edges import TinyEntityEdge
from tiny_graph.graph.multi_layer_graph.edges import TinyEventEdge
from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEventNode
from tiny_graph.graph.multi_layer_graph.ops.cluster_operations import (
    resolve_and_extract_clusters,
)
from tiny_graph.graph.multi_layer_graph.ops.edge_operations import (
    bulk_save_cluster_edges,
)
from tiny_graph.graph.multi_layer_graph.ops.edge_operations import bulk_save_entity_edges
from tiny_graph.graph.multi_layer_graph.ops.edge_operations import bulk_save_event_edges
from tiny_graph.graph.multi_layer_graph.ops.edge_operations import extract_edges
from tiny_graph.graph.multi_layer_graph.ops.edge_operations import (
    resolve_extracted_entity_edges,
)
from tiny_graph.graph.multi_layer_graph.ops.graph_operations import build_indices
from tiny_graph.graph.multi_layer_graph.ops.node_operations import bulk_save_clusters
from tiny_graph.graph.multi_layer_graph.ops.node_operations import bulk_save_entities
from tiny_graph.graph.multi_layer_graph.ops.node_operations import bulk_save_events
from tiny_graph.graph.multi_layer_graph.ops.node_operations import (
    extract_attributes_from_nodes,
)
from tiny_graph.graph.multi_layer_graph.ops.node_operations import (
    resolve_extracted_entity_nodes,
)
from tiny_graph.graph.multi_layer_graph.ops.node_operations import retrieve_events
from tiny_graph.graph.multi_layer_graph.types import DataType
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.graph.multi_layer_graph.utils.custom_types import (
    validate_custom_entity_types,
)
from tiny_graph.helper import generate_uuid
from tiny_graph.helper import get_current_timestamp
from tiny_graph.helper import get_default_subgraph_id
from tiny_graph.helper import parse_timestamp
from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoder
from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.messages import BaseMessage
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.prompt import TinyPrompt
from tinygent.core.runtime.executors import run_in_semaphore
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attributes
from tinygent.core.types.base import TinyModel
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.utils import render_template

logger = logging.getLogger(__name__)


def _get_event_data_type(event: str | dict | BaseMessage) -> DataType:
    match event:
        case str():
            return DataType.TEXT
        case dict():
            return DataType.JSON
        case BaseMessage():
            return DataType.MESSAGE
        case _:
            raise TypeError(f'Unsupported data type: {type(event)}')


class AddRecordResult(TinyModel):
    event: TinyEventNode
    entities: list[TinyEntityNode]
    clusters: list[TinyClusterNode]
    event_edges: list[TinyEventEdge]
    entity_edges: list[TinyEntityEdge]
    cluster_edges: list[TinyClusterEdge]


class EntityExtractorPromptTemplate(TinyPrompt):
    """Used to define prompt template for entity extraction."""

    extract_text: TinyPrompt.UserSystem
    extract_message: TinyPrompt.UserSystem
    extract_json: TinyPrompt.UserSystem
    reflexion: TinyPrompt.UserSystem

    _template_fields = {
        'extract_text.user': {'event_content'},
        'extract_message.user': {'event_content', 'previous_events'},
        'extract_json.user': {'event_content', 'source_description'},
        'reflexion.user': {'event_content', 'previous_events', 'extracted_entities'},
    }


class TinyMultiLayerGraphTemplate(TinyModel):
    """Prompt template for Multi-layer knowledge graph."""

    entity_extractor: EntityExtractorPromptTemplate


class TinyMultiLayerGraph(BaseGraph):
    def __init__(
        self,
        llm: AbstractLLM,
        embedder: AbstractEmbedder,
        cross_encoder: AbstractCrossEncoder,
        driver: BaseDriver,
        prompt_template: TinyMultiLayerGraphTemplate | None = None,
        *,
        last_relevant_events_num: int = 5,
        max_reflexion_iterations_num: int = 4,
    ) -> None:
        super().__init__(
            llm=llm,
            embedder=embedder,
            driver=driver,
        )

        self.clients = TinyGraphClients(
            driver=driver,
            llm=llm,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

        if prompt_template is None:
            from .prompts.default_prompts import get_prompt_template

            prompt_template = get_prompt_template()
        self.prompt_template: TinyMultiLayerGraphTemplate = prompt_template

        self.last_relevant_events_num = last_relevant_events_num
        self.max_reflexion_iterations_num = max_reflexion_iterations_num

    async def build_constraints_and_indices(self):
        await build_indices(self.driver, self.clients)

    @tiny_trace('add_record')
    async def add_record(
        self,
        name: str,
        data: str | dict | BaseMessage,
        description: str,
        *,
        reference_time: datetime | None = None,
        uuid: str | None = None,
        subgraph_id: str | None = None,
        entity_types: dict[str, type[TinyModel]] | None = None,
        edge_types: dict[str, type[TinyModel]] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
        **kwargs,
    ) -> AddRecordResult:
        start = time.time()

        validate_custom_entity_types(entity_types)

        # resolve optional values
        uuid = uuid or generate_uuid()
        subgraph_id = subgraph_id or get_default_subgraph_id()
        reference_time = reference_time or get_current_timestamp()
        entity_types = entity_types or {}

        # Gather last-n previous events
        prev_events = await retrieve_events(
            self.driver,
            reference_time,
            last_n=self.last_relevant_events_num,
            subgraph_ids=[subgraph_id],
        )

        # Create new event
        event = TinyEventNode(
            uuid=uuid,
            name=name,
            description=description,
            subgraph_id=subgraph_id,
            data=data,
            data_type=_get_event_data_type(data),
            valid_at=reference_time,
        )

        # Extract & resolve entities
        entities = self._extract_entities(event, prev_events, entity_types)

        entities, uuid_map = await resolve_extracted_entity_nodes(
            self.clients,
            entities,
            event,
            prev_events,
            entity_types=entity_types,
        )

        # Create default edge type map
        edge_type_map_default = (
            {(NodeType.ENTITY.value, NodeType.ENTITY.value): list(edge_types.keys())}
            if edge_types is not None
            else {(NodeType.ENTITY.value, NodeType.ENTITY.value): []}
        )

        # Extract node attributes
        entities_with_attributes: list[
            TinyEntityNode
        ] = await extract_attributes_from_nodes(
            self.clients.llm,
            self.clients.embedder,
            entities,
            event,
            prev_events,
            entity_types=entity_types,
        )

        # Extract & resolve clusters
        new_clusters, exisiting_clusters = await resolve_and_extract_clusters(
            self.clients,
            entities_with_attributes,
        )

        # Extract & resolve edges
        (
            entity_edges,
            invalidated_entity_edges,
            cluster_edges,
            event_edges,
        ) = await self._extract_edges(
            event,
            prev_events,
            edge_types,
            edge_type_map or edge_type_map_default,
            entities,
            uuid_map,
            new_clusters,
            exisiting_clusters,
            subgraph_id,
        )

        # Process and save data
        await self._bulk_save(
            [event],
            entities_with_attributes,
            new_clusters,
            entity_edges + invalidated_entity_edges,
            cluster_edges,
            event_edges,
        )

        logger.debug('add episode finished in %d seconds', time.time() - start)

        set_tiny_attributes(
            {
                'subgraph_id': subgraph_id,
                'event.uuid': event.uuid,
                'event.content': json.dumps(event.serialized_data),
                'event.valid_at': parse_timestamp(event.valid_at),
                'entity.count': len(entities_with_attributes),
                'entity_edge.count': len(entity_edges),
                'cluster.count': len(new_clusters),
                'cluster_edge.count': len(cluster_edges),
            }
        )

        return AddRecordResult(
            event=event,
            entities=entities_with_attributes,
            clusters=new_clusters,
            event_edges=event_edges,
            entity_edges=entity_edges,
            cluster_edges=cluster_edges,
        )

    @tiny_trace('bulk_save')
    async def _bulk_save(
        self,
        events: list[TinyEventNode] | None = None,
        entities: list[TinyEntityNode] | None = None,
        clusters: list[TinyClusterNode] | None = None,
        entity_edges: list[TinyEntityEdge] | None = None,
        cluster_edges: list[TinyClusterEdge] | None = None,
        event_edges: list[TinyEventEdge] | None = None,
    ) -> None:
        tasks = []
        driver = self.clients.driver

        if events:
            tasks.append(bulk_save_events(driver, events))

        if entities:
            tasks.append(bulk_save_entities(driver, entities))

        if clusters:
            tasks.append(bulk_save_clusters(driver, clusters))

        if entity_edges:
            tasks.append(bulk_save_entity_edges(driver, entity_edges))

        if cluster_edges:
            tasks.append(bulk_save_cluster_edges(driver, cluster_edges))

        if event_edges:
            tasks.append(bulk_save_event_edges(driver, event_edges))

        if tasks:
            await run_in_semaphore(*tasks)

    @tiny_trace('extract_edges')
    async def _extract_edges(
        self,
        current_event: TinyEventNode,
        previous_events: list[TinyEventNode],
        edge_types: dict[str, type[TinyModel]] | None,
        edge_type_map: dict[tuple[str, str], list[str]],
        entities: list[TinyEntityNode],
        entity_uuid_map: dict[str, str],
        extracted_clusters: list[TinyClusterNode],
        existing_clusters: list[TinyClusterNode],
        subgraph_id: str,
    ) -> tuple[
        list[TinyEntityEdge],
        list[TinyEntityEdge],
        list[TinyClusterEdge],
        list[TinyEventEdge],
    ]:
        entity_edges, cluster_edges = await extract_edges(
            self.clients.llm,
            entities,
            current_event,
            previous_events,
            edge_types,
            edge_type_map,
            extracted_clusters,
            existing_clusters,
            subgraph_id,
        )

        (
            resolved_entity_edges,
            invalidated_entity_edges,
        ) = await resolve_extracted_entity_edges(
            self.clients,
            entity_edges,
            current_event,
            entities,
            entity_uuid_map,
            edge_types or {},
            edge_type_map,
        )

        event_edges = [
            TinyEventEdge(
                subgraph_id=subgraph_id,
                source_node_uuid=current_event.uuid,
                target_node_uuid=e.uuid,
            )
            for e in entities
        ]

        return (
            resolved_entity_edges,
            invalidated_entity_edges,
            cluster_edges,
            event_edges,
        )

    @tiny_trace('extract_entities')
    def _extract_entities(
        self,
        current_event: TinyEventNode,
        previous_events: list[TinyEventNode],
        entity_types: dict[str, type[TinyModel]],
    ) -> list[TinyEntityNode]:
        need_revision: bool = True
        reflexion_iteration_count: int = 0
        custom_prompt: str = ''
        extracted_entities: ExtractedEntities | None = None

        entity_types_context = [
            {
                'entity_type_id': 0,
                'entity_type_name': NodeType.ENTITY.value,
                'entity_type_description': 'Default entity classification. Use this entity type if the entity is not one of the other listed types.',
            }
        ]
        entity_types_context.extend(
            [
                {
                    'entity_type_id': i + 1,
                    'entity_type_name': type_name,
                    'entity_type_description': type_model.doc,
                }
                for i, (type_name, type_model) in enumerate(entity_types.items())
            ]
        )

        def _info_data(x: TinyEventNode) -> str | dict:
            if isinstance(x.data, BaseMessage):
                return x.data.tiny_str
            return x.data

        while (
            need_revision
            and reflexion_iteration_count < self.max_reflexion_iterations_num
        ):
            reflexion_iteration_count += 1

            match current_event.data_type:
                case DataType.MESSAGE:
                    prompt = self.prompt_template.entity_extractor.extract_message
                case DataType.TEXT:
                    prompt = self.prompt_template.entity_extractor.extract_message
                case DataType.JSON:
                    prompt = self.prompt_template.entity_extractor.extract_json
                case _:
                    raise ValueError(
                        f'Unknown node datatype: {current_event.data_type}, available types: {", ".join(DataType.__members__)}'
                    )

            extracted_entities = self.llm.generate_structured(
                llm_input=TinyLLMInput(
                    messages=[
                        TinySystemMessage(content=prompt.system),
                        TinyHumanMessage(
                            content=render_template(
                                prompt.user,
                                {
                                    'event_content': _info_data(current_event),
                                    'source_description': current_event.description,
                                    'previous_events': [
                                        _info_data(prev_event)
                                        for prev_event in previous_events
                                    ],
                                    'entity_types': entity_types_context,
                                    'custom_prompt': custom_prompt,
                                },
                            )
                        ),
                    ]
                ),
                output_schema=ExtractedEntities,
            )

            logger.debug(
                'Extracted (%d) entities: %s',
                len(extracted_entities.extracted_entities),
                [e.name for e in extracted_entities.extracted_entities],
            )

            missed_entities = self.llm.generate_structured(
                llm_input=TinyLLMInput(
                    messages=[
                        TinySystemMessage(
                            content=self.prompt_template.entity_extractor.reflexion.system
                        ),
                        TinyHumanMessage(
                            content=render_template(
                                self.prompt_template.entity_extractor.reflexion.user,
                                {
                                    'event_content': _info_data(current_event),
                                    'previous_events': [
                                        _info_data(prev_event)
                                        for prev_event in previous_events
                                    ],
                                    'extracted_entities': [
                                        e.name
                                        for e in extracted_entities.extracted_entities
                                    ],
                                    'custom_prompt': custom_prompt,
                                },
                            )
                        ),
                    ]
                ),
                output_schema=MissedEntities,
            )

            need_revision = len(missed_entities.missed_entities) > 0

            logger.debug(
                'Entities extraction %s revision',
                'need' if need_revision else "don't need",
            )
            if need_revision:
                logger.debug(
                    'Entities reflexion missed entities: %s',
                    missed_entities.missed_entities,
                )

            custom_prompt = f'Make sure that the following entities are extracted: {"\n".join(missed_entities.missed_entities)}'
        if not extracted_entities:
            logger.warning('No entities extracted.')
            return []

        extracted_entity_nodes: list[TinyEntityNode] = []
        extracted_entities_proc: list[ExtractedEntity] = [
            e for e in extracted_entities.extracted_entities
        ]

        for extracted_entity in extracted_entities_proc:
            entity_type_name = next(
                (
                    e.get('entity_type_name')
                    for e in entity_types_context
                    if e.get('entity_type_id') == extracted_entity.entity_type_id
                ),
                NodeType.ENTITY.value,
            )

            labels: list[str] = list({NodeType.ENTITY.value, str(entity_type_name)})
            new_entity = TinyEntityNode(
                subgraph_id=current_event.subgraph_id,
                name=extracted_entity.name,
                labels=labels,
                summary='',
            )
            extracted_entity_nodes.append(new_entity)

        return extracted_entity_nodes

    async def close(self) -> None:
        await self.driver.close()
