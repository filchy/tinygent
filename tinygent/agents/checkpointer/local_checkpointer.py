from __future__ import annotations

from copy import deepcopy
import logging
from threading import Lock
from typing import Any
from typing import TypeVar

from pydantic import Field

from tinygent.agents.checkpointer.base_checkpointer import TinyBaseCheckpointer
from tinygent.agents.checkpointer.base_checkpointer import TinyBaseCheckpointerConfig

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='TinyLocalCheckpointer')

_GLOBAL_CHECKPOINTS: dict[str, Any] = {}
_GLOBAL_CHECKPOINTS_LOCK = Lock()


class TinyLocalCheckpointerConfig(TinyBaseCheckpointerConfig['TinyLocalCheckpointer']):
    type: Any = Field(default='local', frozen=True)

    def build(self) -> TinyLocalCheckpointer:
        return TinyLocalCheckpointer(self.data)


class TinyLocalCheckpointer(TinyBaseCheckpointer):
    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(data)

    def save(self, checkpoint_id: str) -> None:
        logger.debug('Saving checkpoint %s', checkpoint_id)
        with _GLOBAL_CHECKPOINTS_LOCK:
            _GLOBAL_CHECKPOINTS[checkpoint_id] = deepcopy(self.data)
            logger.debug('Data %s saved: %s', checkpoint_id, str(self.data))

    def load(self, checkpoint_id: str) -> None:
        logger.debug('Loading checkpoint %s', checkpoint_id)
        with _GLOBAL_CHECKPOINTS_LOCK:
            tmp = _GLOBAL_CHECKPOINTS.get(checkpoint_id)
            if tmp is None:
                logger.warning("Couldn't find data for checkpoint id: %s", checkpoint_id)
                return
            self.data = deepcopy(tmp)
            logger.debug('Data %s loaded: %s', checkpoint_id, str(self.data))

    def delete(self, checkpoint_id: str) -> None:
        logger.debug('Deleting checkpoint %s', checkpoint_id)
        with _GLOBAL_CHECKPOINTS_LOCK:
            _GLOBAL_CHECKPOINTS.pop(checkpoint_id)
