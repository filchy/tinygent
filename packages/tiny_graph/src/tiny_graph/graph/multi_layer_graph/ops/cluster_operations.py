from tinygent.datamodels.embedder import AbstractEmbedder
from tinygent.datamodels.llm import AbstractLLM
from tinygent.runtime.executors import run_in_semaphore

from tiny_graph.driver.base import BaseDriver
from tiny_graph.graph.multi_layer_graph.nodes import TinyClusterNode
from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tiny_graph.graph.multi_layer_graph.types import EdgeType
from tiny_graph.graph.multi_layer_graph.types import NodeType


async def determine_entity_cluster(
    driver: BaseDriver,
    entity: TinyEntityNode,
) -> tuple[TinyClusterNode | None, bool]:
    records, _, _ = await driver.execute_query(
        f'''
        MATCH (c:{NodeType.CLUSTER.value})-[:{EdgeType.HAS_MEMBER.value}]->(n:{NodeType.ENTITY.value} {{ uuid: $entity_uuid }})
        RETURN
            c.uuid AS uuid,
            c.name AS name,
            c.subgraph_id AS subgraph_id,
            c.created_at AS created_at,
            c.name_embedding AS name_embedding,
            c.summary AS summary
        '''
    )
    return None, False


async def resolve_and_extract_cluster(
    driver: BaseDriver,
    llm: AbstractLLM,
    embedder: AbstractEmbedder,
    entity: TinyEntityNode,
) -> TinyClusterNode | None:
    cluster, is_new = await determine_entity_cluster(driver, entity)


async def resolve_and_extract_clusters(
    driver: BaseDriver,
    llm: AbstractLLM,
    embedder: AbstractEmbedder,
    entities: list[TinyEntityNode],
) -> list[TinyClusterNode]:
    results = await run_in_semaphore(
        *[
            resolve_and_extract_cluster(driver, llm, embedder, entity)
            for entity in entities
        ]
    )

    return [r for r in results if r]
