import logging

from tiny_graph.driver.base import BaseDriver
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.queries.graph_queries import (
    build_indices_and_constraints,
)
from tiny_graph.types.provider import GraphProvider

logger = logging.getLogger(__name__)


async def build_indices(driver: BaseDriver, clients: TinyGraphClients):
    provider = driver.provider
    queries = build_indices_and_constraints(provider, clients)

    if provider == GraphProvider.NEO4J:
        from neo4j.exceptions import ClientError

        for query in queries:
            try:
                await driver.execute_query(query)
            except ClientError as e:
                logger.warning('neo4j error while building indices: %s', e)
                continue

        return

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )
