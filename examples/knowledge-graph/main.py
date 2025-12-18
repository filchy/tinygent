from datetime import datetime
from datetime import timezone
import os
from tiny_graph import TinyMultiLayerGraph
from tiny_graph.driver import Neo4jDriver
from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.search.search import search
from tiny_graph.graph.multi_layer_graph.search.search_cfg import TinyEntitySearchConfig, TinySearchConfig, EntitySearchMethods

from tinygent.factory.embedder import build_embedder
from tinygent.factory.llm import build_llm
from tinygent.logging import setup_logger

logger = setup_logger('debug')


neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')


async def main():
    driver = Neo4jDriver(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )
    await driver.health_check()

    llm = build_llm('openai:gpt-4o')
    embedder = build_embedder('openai:text-embedding-3-small')

    all_clients = TinyGraphClients(
        driver=driver,
        llm=llm,
        embedder=embedder,
    )

    graph = TinyMultiLayerGraph(
        llm=build_llm('openai:gpt-4o'),
        embedder=build_embedder('openai:text-embedding-3-small'),
        driver=driver,
    )
    await graph.build_constraints_and_indices()

    results = await search(
        'What is sleeper agent?',
        clients=all_clients,
        config=TinySearchConfig(
            entity_search=TinyEntitySearchConfig(
                search_methods=[
                    EntitySearchMethods.COSINE_SIM,
                    EntitySearchMethods.BM_25
                ]
            )
        )
    )
    logger.info(results)

    # texts = [
    #     {
    #         'name': 'Agent Raven',
    #         'description': 'Double agent active in early Cold War intelligence operations',
    #         'text': 'Agent Raven operated as a double agent during the early Cold War, passing controlled information between Eastern and Western intelligence services.'
    #     },
    #     {
    #         'name': 'Operation Silent Pen',
    #         'description': 'Undercover diplomatic infiltration mission',
    #         'text': 'In 1952, an undercover intelligence agent infiltrated a diplomatic mission to gather information about nuclear negotiations.'
    #     },
    #     {
    #         'name': 'Sleeper Asset Echo',
    #         'description': 'Long-term sleeper agent activated during crisis',
    #         'text': 'A sleeper agent was activated after several years of inactivity to influence political decisions during a Cold War crisis.'
    #     },
    #     {
    #         'name': 'SIGINT Unit North',
    #         'description': 'Signals intelligence group monitoring enemy communications',
    #         'text': 'Signals intelligence agents monitored encrypted radio transmissions to track troop movements behind the Iron Curtain.'
    #     },
    #     {
    #         'name': 'Handler Atlas',
    #         'description': 'Senior intelligence handler coordinating field agents',
    #         'text': 'An intelligence handler managed multiple field agents, coordinating dead drops and coded messages throughout Europe.'
    #     }
    # ]
    #
    # for text in texts:
    #     await graph.add_record(
    #         text['name'],
    #         text['text'],
    #         text['description'],
    #         reference_time=datetime.now(timezone.utc)
    #     )
    #
    # await graph.close()


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
