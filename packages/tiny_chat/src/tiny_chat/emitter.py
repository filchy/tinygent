from __future__ import annotations

import typing
from typing import Awaitable
from typing import Callable

if typing.TYPE_CHECKING:
    from tiny_chat.message import BaseMessage


class _Emitter:
    _send: Callable[[BaseMessage], Awaitable[None]] | None = None

    def configure(self, sender: Callable[[BaseMessage], Awaitable[None]]):
        self._send = sender

    async def send(self, msg: BaseMessage):
        if not self._send:
            raise RuntimeError('Emitter not configured')
        await self._send(msg)


emitter = _Emitter()
