import logging
from typing import Any
from typing import cast

from neo4j import AsyncGraphDatabase
from neo4j import EagerResult
from typing_extensions import LiteralString

from tiny_graph.driver.base import BaseDriver
from tiny_graph.types.provider import GraphProvider

logger = logging.getLogger(__name__)


class Neo4jDriver(BaseDriver):
    provider = GraphProvider.NEO4J

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password

        self.__client = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
        )

    async def health_check(self) -> None:
        try:
            await self.__client.verify_connectivity()
            logger.debug('Neo4j connection is healthy')
        except Exception as e:
            logger.error('Neo4j health check failed: %s', e)
            raise e

    async def execute_query(
        self, query: str | LiteralString, **kwargs: Any
    ) -> EagerResult:
        params = kwargs.pop('params', {})

        if isinstance(query, str):
            query = cast(LiteralString, query)

        try:
            result = await self.__client.execute_query(
                query, parameters_=params, **kwargs
            )
        except Exception as e:
            logger.error('Neo4j failed to execute query: %s with error: %s', query, e)
            raise e
        return result

    async def close(self) -> None:
        await self.__client.close()
        logger.debug('Neo4j connection closed')
