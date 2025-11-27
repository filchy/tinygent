from contextvars import ContextVar

current_chat_id: ContextVar[str] = ContextVar('current_chat_id')
