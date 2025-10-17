import json
import os
from typing import IO
from typing import Any
from typing import cast

import yaml


class TinySafeLoader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def _construct_include(safe_loader: TinySafeLoader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    scalar = cast(str, safe_loader.construct_scalar(cast(yaml.ScalarNode, node)))
    filename = os.path.join(safe_loader._root, scalar)
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            loaded_data = yaml.load(f, TinySafeLoader)
        elif extension in ('json',):
            loaded_data = json.load(f)
        else:
            loaded_data = ''.join(f.readlines())

    return loaded_data


yaml.add_constructor('!include', _construct_include, TinySafeLoader)


def tiny_yaml_load(yaml_path: str) -> Any:
    """Load a YAML file with `!include` support."""
    with open(yaml_path, 'r') as f:
        return yaml.load(f, TinySafeLoader)


if __name__ == '__main__':
    data = tiny_yaml_load('foo.yaml')
    print(data)
