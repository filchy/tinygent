from __future__ import annotations

import logging
from typing import Any

from tinygent.agents.checkpointer.base_checkpointer import TinyBaseCheckpointer

logger = logging.getLogger(__name__)


class TinyDefaultCheckpointer(TinyBaseCheckpointer):
    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(data)

    def save(self, checkpoint_id: str) -> None:
        logger.debug(
            'Checkpoint (%s) will not be saved in default checkpointer', checkpoint_id
        )

    def load(self, checkpoint_id: str) -> None:
        logger.debug('Nothing to be loaded for default checkpointer (%s)', checkpoint_id)

    def delete(self, checkpoint_id: str) -> None:
        logger.debug('Deleting checkpoint %s', checkpoint_id)
        self.data = {}
