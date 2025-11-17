__all__ = [
    'TinyModel',
    'TinyModelBuildable',
    'TinyPromptTemplate',
    'HasDiscriminatorField',
]


def __getattr__(name: str):
    if name == 'TinyModel':
        from tinygent.types.base import TinyModel

        return TinyModel
    elif name == 'TinyModelBuildable':
        from tinygent.types.builder import TinyModelBuildable

        return TinyModelBuildable
    elif name == 'TinyPromptTemplate':
        from tinygent.types.prompt_template import TinyPromptTemplate

        return TinyPromptTemplate
    elif name == 'HasDiscriminatorField':
        from tinygent.types.discriminator import HasDiscriminatorField

        return HasDiscriminatorField
    else:
        raise AttributeError(f'module {__name__} has no attribute {name}')
