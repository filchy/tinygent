from tiny_graph.graph.multi_layer_graph.nodes import TinyEntityNode
from tinygent.core.types.base import TinyModel


def validate_custom_entity_types(
    entity_types: dict[str, type[TinyModel]] | None,
) -> None:
    if not entity_types:
        return

    original_entity_fiels = set(TinyEntityNode.model_fields)

    for type_name, type_model in entity_types.items():
        custom_fields = set(type_model.model_fields)
        if overlap := original_entity_fiels & custom_fields:
            raise ValueError(
                f'Custom entity type "{type_name}" defines fields '
                f'that collide with TinyEntityNode: {sorted(overlap)}'
            )
    return
