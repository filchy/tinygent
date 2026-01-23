__all__ = [
    'TinyModel',
    'TinyModelBuildable',
    'TinyPromptTemplate',
    'HasDiscriminatorField',
]


def __getattr__(name: str):
    if name == 'TinyModel':
        from tinygent.core.types.base import TinyModel

        return TinyModel
    elif name == 'TinyModelBuildable':
        from tinygent.core.types.builder import TinyModelBuildable

        return TinyModelBuildable
    elif name == 'TinyPromptTemplate':
        from tinygent.core.types.prompt_template import TinyPromptTemplate

        return TinyPromptTemplate
    elif name == 'HasDiscriminatorField':
        from tinygent.core.types.discriminator import HasDiscriminatorField

        return HasDiscriminatorField
    else:
        raise AttributeError(f'module {__name__} has no attribute {name}')
