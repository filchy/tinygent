from typing import Any

from .context import current_session


class SessionProxy:
    @property
    def _session(self):
        session = current_session.get()
        if session is None:
            raise RuntimeError('No active session found.')
        return session

    def set(self, key: str, value: Any) -> None:
        self._session.set(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._session.get(key, default)

    def has(self, key: str) -> bool:
        return self._session.has(key)
