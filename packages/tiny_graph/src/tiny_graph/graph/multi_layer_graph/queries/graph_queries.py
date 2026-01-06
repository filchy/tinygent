from tiny_graph.graph.multi_layer_graph.datamodels.clients import TinyGraphClients
from tiny_graph.graph.multi_layer_graph.types import NodeType
from tiny_graph.types.provider import GraphProvider


def build_indices_and_constraints(
    provider: GraphProvider, clients: TinyGraphClients
) -> list[str]:
    return (
        get_constraints(provider)
        + get_fulltext_indices(provider)
        + get_vector_indices(provider, clients)
    )


def get_constraints(
    provider: GraphProvider,
) -> list[str]:
    if provider == GraphProvider.NEO4J:
        return [
            f"""
            CREATE CONSTRAINT {NodeType.EVENT.value}_uuid_unique IF NOT EXISTS
            FOR (e:{NodeType.EVENT.value})
            REQUIRE e.uuid IS UNIQUE;
            """,
            f"""
            CREATE CONSTRAINT {NodeType.ENTITY.value}_uuid_unique IF NOT EXISTS
            FOR (e:{NodeType.ENTITY.value})
            REQUIRE e.uuid IS UNIQUE;
            """,
            f"""
            CREATE CONSTRAINT {NodeType.CLUSTER.value}_uuid_unique IF NOT EXISTS
            FOR (c:{NodeType.CLUSTER.value})
            REQUIRE c.uuid IS UNIQUE;
            """,
        ]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def get_fulltext_indices(
    provider: GraphProvider,
) -> list[str]:
    if provider == GraphProvider.NEO4J:
        return [
            f"""
            CREATE FULLTEXT INDEX `{NodeType.ENTITY.value}_fulltext_index`
            IF NOT EXISTS
            FOR (e:{NodeType.ENTITY.value})
            ON EACH [
                e.name,
                e.summary
            ];
            """,
            f"""
            CREATE FULLTEXT INDEX `{NodeType.CLUSTER.value}_fulltext_index`
            IF NOT EXISTS
            FOR (e:{NodeType.CLUSTER.value})
            ON EACH [
                e.name,
                e.summary
            ];
            """,
            """
            CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS
            FOR ()-[e:RELATES_TO]-()
            ON EACH [
                e.name,
                e.fact
            ]
            """,
        ]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )


def get_vector_indices(
    provider: GraphProvider,
    clients: TinyGraphClients,
) -> list[str]:
    if provider == GraphProvider.NEO4J:
        return [
            f"""
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
            """,
            f"""
            CREATE VECTOR INDEX `{NodeType.CLUSTER.value}_{clients.safe_embed_model}_name_embedding_index`
            IF NOT EXISTS
            FOR (e:{NodeType.CLUSTER.value})
            ON (e.name_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {clients.embedder.embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }};
            """,
            f"""
            CREATE VECTOR INDEX `RELATES_TO_{clients.safe_embed_model}_fact_embedding_index`
            IF NOT EXISTS
            FOR ()-[r:RELATES_TO]-()
            ON (r.fact_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {clients.embedder.embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }};
            """,
        ]

    raise ValueError(
        f'Unknown provider was given: {provider}, available providers: {", ".join(provider.__members__)}'
    )
