from typing import Protocol


class HasDiscriminatorField(Protocol):
    @classmethod
    def get_discriminator_field(cls) -> str: ...
