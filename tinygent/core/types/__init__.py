__all__ = [
    'TinyModel',
    'TinyModelBuildable',
    'TinyPrompt',
    'HasDiscriminatorField',
]


def __getattr__(name: str):
    if name == 'TinyModel':
        from tinygent.core.types.base import TinyModel

        return TinyModel
    elif name == 'TinyModelBuildable':
        from tinygent.core.types.builder import TinyModelBuildable

        return TinyModelBuildable
    elif name == 'TinyPrompt':
        from tinygent.core.prompt import TinyPrompt

        return TinyPrompt
    elif name == 'HasDiscriminatorField':
        from tinygent.core.types.discriminator import HasDiscriminatorField

        return HasDiscriminatorField
    else:
        raise AttributeError(f'module {__name__} has no attribute {name}')
