def _register_voyageai() -> None:
    from tinygent.core.runtime.global_registry import GlobalRegistry

    from .cross_encoder import VoyageAICrossEncoder
    from .cross_encoder import VoyageAICrossEncoderConfig
    from .embedder import VoyageAIEmbedder
    from .embedder import VoyageAIEmbedderConfig

    registry = GlobalRegistry().get_registry()

    registry.register_crossencoder(
        'voyageai', VoyageAICrossEncoderConfig, VoyageAICrossEncoder
    )
    registry.register_embedder('voyageai', VoyageAIEmbedderConfig, VoyageAIEmbedder)


_register_voyageai()
