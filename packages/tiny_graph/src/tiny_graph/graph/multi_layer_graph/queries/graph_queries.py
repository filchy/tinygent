from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.types.provider import GraphProvider


def build_indices_and_constraints(
    provider: GraphProvider,
    clients: TinyGraphClients
) -> list[str]:
    if provider == GraphProvider.NEO4J:
        return [
            f'''
            CREATE CONSTRAINT {NodeType.EVENT.value}_uuid_unique IF NOT EXISTS
            FOR (e:{NodeType.EVENT.value})
            REQUIRE e.uuid IS UNIQUE;
            ''',

            f'''
            CREATE CONSTRAINT {NodeType.ENTITY.value}_uuid_unique IF NOT EXISTS
            FOR (e:{NodeType.ENTITY.value})
            REQUIRE e.uuid IS UNIQUE;
            ''',

            f'''
            CREATE CONSTRAINT {NodeType.CLUSTER.value}_uuid_unique IF NOT EXISTS
            FOR (c:{NodeType.CLUSTER.value})
            REQUIRE c.uuid IS UNIQUE;
            ''',

            f'''
            CREATE VECTOR INDEX `{NodeType.ENTITY.value}_{clients.safe_embed_model}_name_embedding_index`
            IF NOT EXISTS
            FOR (e:{NodeType.ENTITY.value})
            ON (e.name_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {clients.embedder.embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }};
            ''',

            f'''
            CREATE FULLTEXT INDEX `{NodeType.ENTITY.value}_fulltext_index`
            IF NOT EXISTS
            FOR (e:{NodeType.ENTITY.value})
            ON EACH [
                e.name,
                e.summary
            ]
            ''',
        ]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )
