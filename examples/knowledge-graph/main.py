from datetime import datetime
from datetime import timezone
import os

from tiny_graph import TinyMultiLayerGraph
from tiny_graph.driver import Neo4jDriver
from tiny_graph.graph.multi_layer_graph import search
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.search.search_presets import (
    NODE_HYBRID_SEARCH_RRF,
)
from tinygent.core.factory import build_cross_encoder
from tinygent.core.factory import build_embedder
from tinygent.core.factory import build_llm
from tinygent.core.types import TinyModel
from tinygent.logging import setup_logger

logger = setup_logger('debug')

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')


# Custom entity types
class Person(TinyModel):
    """A person entity representing an individual mentioned in intelligence records."""

    person_name: str


class Place(TinyModel):
    """A geographical location or place mentioned in intelligence records."""

    place_name: str


class Organization(TinyModel):
    """An organization, agency, or group mentioned in intelligence records."""

    organization_name: str


class Operation(TinyModel):
    """A covert operation or mission mentioned in intelligence records."""

    operation_name: str


# Custom edge types
class WorksFor(TinyModel):
    """Relationship indicating an agent works for an organization."""

    role: str


class LocatedIn(TinyModel):
    """Relationship indicating something or someone is located in a place."""

    time_period: str


class ParticipatedIn(TinyModel):
    """Relationship indicating a person participated in an operation."""

    role: str


class Knows(TinyModel):
    """Relationship indicating two people know each other."""

    relationship_type: str


ENTITY_TYPES = {
    'Person': Person,
    'Place': Place,
    'Organization': Organization,
    'Operation': Operation,
}

EDGE_TYPES = {
    'WORKS_FOR': WorksFor,
    'LOCATED_IN': LocatedIn,
    'PARTICIPATED_IN': ParticipatedIn,
    'KNOWS': Knows,
}

# Map which edge types can connect which entity types
EDGE_TYPE_MAP = {
    ('Person', 'Organization'): ['WORKS_FOR'],
    ('Person', 'Place'): ['LOCATED_IN'],
    ('Organization', 'Place'): ['LOCATED_IN'],
    ('Person', 'Operation'): ['PARTICIPATED_IN'],
    ('Person', 'Person'): ['KNOWS'],
    ('Entity', 'Entity'): ['WORKS_FOR', 'LOCATED_IN', 'PARTICIPATED_IN', 'KNOWS'],
}


async def main():
    driver = Neo4jDriver(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )
    await driver.health_check()

    llm = build_llm('openai:gpt-4o-mini')
    embedder = build_embedder('openai:text-embedding-3-small')
    cross_encoder = build_cross_encoder('llm', llm='openai:gpt-4o-mini')

    graph = TinyMultiLayerGraph(
        llm=llm,
        embedder=embedder,
        cross_encoder=cross_encoder,
        driver=driver,
    )
    await graph.build_constraints_and_indices()

    # insert data
    texts = [
        {
            'name': 'Agent Raven',
            'description': 'Double agent active in early Cold War intelligence operations',
            'text': 'Agent Raven operated as a double agent during the early Cold War, passing controlled information between Eastern and Western intelligence services. He was based in Berlin and worked for the CIA.',
        },
        {
            'name': 'Operation Silent Pen',
            'description': 'Undercover diplomatic infiltration mission',
            'text': 'In 1952, Agent Raven and Agent Echo infiltrated a diplomatic mission in Vienna to gather information about nuclear negotiations. The operation was coordinated by the CIA.',
        },
        {
            'name': 'Sleeper Asset Echo',
            'description': 'Long-term sleeper agent activated during crisis',
            'text': 'Agent Echo was a sleeper agent activated after several years of inactivity to influence political decisions during a Cold War crisis. She knew Agent Raven from their training in London.',
        },
        {
            'name': 'SIGINT Unit North',
            'description': 'Signals intelligence group monitoring enemy communications',
            'text': 'SIGINT Unit North, a division of the NSA, monitored encrypted radio transmissions from their base in Frankfurt to track Soviet troop movements behind the Iron Curtain.',
        },
        {
            'name': 'Handler Atlas',
            'description': 'Senior intelligence handler coordinating field agents',
            'text': 'Handler Atlas managed Agent Raven and Agent Echo from the MI6 headquarters in London, coordinating dead drops and coded messages throughout Europe.',
        },
    ]

    for text in texts:
        await graph.add_record(
            text['name'],
            text['text'],
            text['description'],
            reference_time=datetime.now(timezone.utc),
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )

    # search in knowledge graph
    clients = TinyGraphClients(
        driver=driver,
        llm=llm,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    search_results = await search(
        query='agent raven',
        clients=clients,
        config=NODE_HYBRID_SEARCH_RRF,
    )

    logger.info(
        'results [%d]: %s', len(search_results.entities), search_results.entities
    )

    await graph.close()


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
