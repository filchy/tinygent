from datetime import datetime
import logging

from tinygent.datamodels.embedder import AbstractEmbedder
from tinygent.datamodels.llm import AbstractLLM
from tinygent.telemetry.decorators import tiny_trace
from tinygent.datamodels.messages import BaseMessage
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinySystemMessage
from tinygent.types.base import TinyModel
from tinygent.types.io.llm_io_input import TinyLLMInput
from tinygent.types.prompt_template import TinyPromptTemplate
from tinygent.utils import render_template

from tiny_graph.driver.base import BaseDriver
from tiny_graph.graph.base import BaseGraph
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.datamodels.extract_nodes import ExtractedEntities
from tiny_graph.graph.multi_layer_graph.datamodels.extract_nodes import ExtractedEntity
from tiny_graph.graph.multi_layer_graph.datamodels.extract_nodes import MissedEntities
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEventNode
from tiny_graph.graph.multi_layer_graph.ops.graph_operations import build_indices
from tiny_graph.graph.multi_layer_graph.ops.node_operations import retrieve_events
from tiny_graph.graph.multi_layer_graph.queries.node_queries import get_last_n_event_nodes
from tiny_graph.graph.multi_layer_graph.types import DataType
from tiny_graph.helper import generate_uuid
from tiny_graph.helper import get_default_subgraph_id

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


class EntityExtractorPromptTemplate(TinyPromptTemplate):
    """Used to define prompt template for entity extraction."""

    extract_text: TinyPromptTemplate.UserSystem
    extract_message: TinyPromptTemplate.UserSystem
    extract_json: TinyPromptTemplate.UserSystem
    reflexion: TinyPromptTemplate.UserSystem

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
        reference_time: datetime,
        uuid: str | None = None,
        subgraph_id: str | None = None,
        entity_types: dict[str, type[TinyModel]] = {},
        **kwargs
    ) -> None:
        uuid = uuid or generate_uuid()
        subgraph_id = subgraph_id or get_default_subgraph_id()

        prev_events = await retrieve_events(
            self.driver,
            reference_time,
            last_n=self.last_relevant_events_num,
            subgraph_ids=[subgraph_id]
        )
        logger.info('prev event: %s', prev_events)

        event = TinyEventNode(
            uuid=uuid,
            name=name,
            description=description,
            subgraph_id=subgraph_id,
            data=data,
            data_type=_get_event_data_type(data),
            valid_at=reference_time,
        )
        logger.info('event: %s', event)

        entities = self._extract_entities(event, prev_events, entity_types)
        logger.info('extracted entities: %s', entities)

        self._deduplicate_extracted_nodes(entities, event, prev_events)

    @tiny_trace('deduplicate_extracted_nodes')
    def _deduplicate_extracted_nodes(
        self,
        extracted_entities: list[TinyEntityNode],
        current_event: TinyEventNode,
        previous_events: list[TinyEventNode],
        entity_types: dict[str, type[TinyModel]] = {},
    ):
        pass

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
                'entity_type_name': 'Entity',
                'entity_type_description': 'Default entity classification. Use this entity type if the entity is not one of the other listed types.',
            }
        ]
        entity_types_context.extend([
            {
                'entity_type_id': i + 1,
                'entity_type_name': type_name,
                'entity_type_description': type_model.__doc__ or 'No description provided'
            } for i, (type_name, type_model) in enumerate(entity_types.items())
        ])

        def _info_data(x: TinyEventNode) -> str | dict:
            if isinstance(x.data, BaseMessage):
                return x.data.tiny_str
            return x.data

        while need_revision and reflexion_iteration_count < self.max_reflexion_iterations_num:
            reflexion_iteration_count += 1

            match current_event.data_type:
                case DataType.MESSAGE:
                    prompt = self.prompt_template.entity_extractor.extract_message
                case DataType.TEXT:
                    prompt = self.prompt_template.entity_extractor.extract_message
                case DataType.JSON:
                    prompt = self.prompt_template.entity_extractor.extract_json
                case _:
                    raise ValueError(f'Unknown node datatype: {current_event.data_type}, available types: {', '.join(DataType.__members__)}')

            extracted_entities = self.llm.generate_structured(
                llm_input=TinyLLMInput(
                    messages=[
                        TinySystemMessage(content=prompt.system),
                        TinyHumanMessage(content=render_template(
                            prompt.user,
                            {
                                'event_content': _info_data(current_event),
                                'source_description': current_event.description,
                                'previous_events': [_info_data(prev_event) for prev_event in previous_events],
                                'entity_types': entity_types_context,
                                'custom_prompt': custom_prompt,
                            },
                        )),
                    ]
                ),
                output_schema=ExtractedEntities,
            )

            missed_entities = self.llm.generate_structured(
                llm_input=TinyLLMInput(
                    messages=[
                        TinySystemMessage(content=self.prompt_template.entity_extractor.reflexion.system),
                        TinyHumanMessage(content=render_template(
                            self.prompt_template.entity_extractor.reflexion.user,
                            {
                                'event_content': _info_data(current_event),
                                'previous_events': [_info_data(prev_event) for prev_event in previous_events],
                                'extracted_entities': [e.name for e in extracted_entities.extracted_entities],
                                'custom_prompt': custom_prompt,
                            },
                        )),
                    ]
                ),
                output_schema=MissedEntities,
            )

            need_revision = len(missed_entities.missed_entities) > 0

            custom_prompt = f'Make sure that the following entities are extracted: {'\n'.join(missed_entities.missed_entities)}'
        if not extracted_entities:
            logger.warning('No entities extracted.')
            return []

        extracted_entity_nodes: list[TinyEntityNode] = []
        extracted_entities_proc: list[ExtractedEntity] = [e for e in extracted_entities.extracted_entities]

        for extracted_entity in extracted_entities_proc:
            entity_type_name = next(
                (
                    e.get('entity_type_name')
                    for e in entity_types_context
                    if e.get('entity_type_id') == extracted_entity.entity_type_id
                ), 'Entity'
            )

            labels: list[str] = list({'Entity', str(entity_type_name)})
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
