from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEventNode


def event_node_2_prompt(
    event: TinyEventNode | list[TinyEventNode],
) -> dict | list[dict]:
    if isinstance(event, TinyEventNode):
        return event.model_dump(mode='json')

    return [e.model_dump(mode='json') for e in event]


def entity_node_2_prompt(
    entity: TinyEntityNode | list[TinyEntityNode],
) -> dict | list[dict]:
    if isinstance(entity, TinyEntityNode):
        return entity.model_dump(mode='json')

    return [e.model_dump(mode='json') for e in entity]
