from pydantic import Field

from tinygent.core.types.base import TinyModel


class ExtractedEntity(TinyModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )


class ExtractedEntities(TinyModel):
    extracted_entities: list[ExtractedEntity] = Field(
        ..., description='List of extracted entities'
    )


class MissedEntities(TinyModel):
    missed_entities: list[str] = Field(
        ..., description="Names of entities that weren't extracted"
    )
