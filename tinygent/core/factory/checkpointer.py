from tinygent.core.datamodels.checkpointer import AbstractCheckpointer
from tinygent.core.datamodels.checkpointer import AbstractCheckpointerConfig
from tinygent.core.factory.helper import check_modules
from tinygent.core.factory.helper import parse_config
from tinygent.core.runtime.global_registry import GlobalRegistry


def build_checkpointer(
    checkpointer: str | dict | AbstractCheckpointer | AbstractCheckpointerConfig,
    **kwargs,
) -> AbstractCheckpointer:
    """Build tiny checkpointer."""
    if isinstance(checkpointer, AbstractCheckpointer):
        return checkpointer

    check_modules()

    if isinstance(checkpointer, str):
        checkpointer = {'type': checkpointer, **kwargs}

    if isinstance(checkpointer, AbstractCheckpointerConfig):
        checkpointer = checkpointer.model_dump()

    checkpointer_config = parse_config(
        checkpointer, lambda: GlobalRegistry.get_registry().get_checkpointers()
    )
    return checkpointer_config.build()
