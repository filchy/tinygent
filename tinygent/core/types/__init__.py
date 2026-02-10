from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import TinyModel
    from .builder import TinyModelBuildable
    from .discriminator import HasDiscriminatorField
    from .io.llm_io_chunks import TinyLLMResultChunk
    from .io.llm_io_input import TinyLLMInput
    from .io.llm_io_result import TinyLLMResult

__all__ = [
    'TinyLLMResultChunk',
    'TinyLLMInput',
    'TinyLLMResult',
    'TinyModel',
    'TinyModelBuildable',
    'HasDiscriminatorField',
]


def __getattr__(name: str):
    if name == 'TinyLLMResultChunk':
        from .io.llm_io_chunks import TinyLLMResultChunk

        return TinyLLMResultChunk

    if name == 'TinyLLMInput':
        from .io.llm_io_input import TinyLLMInput

        return TinyLLMInput

    if name == 'TinyLLMResult':
        from .io.llm_io_result import TinyLLMResult

        return TinyLLMResult

    if name == 'TinyModel':
        from .base import TinyModel

        return TinyModel

    if name == 'TinyModelBuildable':
        from .builder import TinyModelBuildable

        return TinyModelBuildable

    if name == 'HasDiscriminatorField':
        from .discriminator import HasDiscriminatorField

        return HasDiscriminatorField

    raise AttributeError(name)
