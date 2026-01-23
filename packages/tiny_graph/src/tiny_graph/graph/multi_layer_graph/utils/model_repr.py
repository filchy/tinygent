from tinygent.core.types.base import TinyModel


def compact_model_repr(model: TinyModel) -> str:
    data = model.model_dump(mode='json')

    for k, v in data.items():
        if k.endswith('_embedding') and isinstance(v, list):
            if len(v) > 6:
                data[k] = v[:3] + ['...'] + v[-3:]

    items = ', '.join(f'{k}={v!r}' for k, v in data.items())
    return f'{model.__class__.__name__}({items})'
