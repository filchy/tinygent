from datetime import datetime

from tiny_graph.driver.base import BaseDriver
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEventNode
from tiny_graph.graph.multi_layer_graph.queries.node_queries import get_last_n_event_nodes
from tiny_graph.types.provider import GraphProvider


async def _find_entity_duplicite_candidates(
    clients: TinyGraphClients,
    extracted_entities: TinyEntityNode,
):
    search_result = []


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
):
    pass
