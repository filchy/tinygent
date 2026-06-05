from tinygent.agents.checkpointer.local_checkpointer import TinyLocalCheckpointer
from tinygent.agents.checkpointer.local_checkpointer import TinyLocalCheckpointerConfig
from tinygent.core.runtime.global_registry import GlobalRegistry


def _register_checkpointers() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_checkpointer(
        'local', TinyLocalCheckpointerConfig, TinyLocalCheckpointer
    )


_register_checkpointers()
