from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.memory.buffer_chat_memory import BufferChatMemoryConfig
from tinygent.memory.buffer_window_chat_memory import BufferWindowChatMemory
from tinygent.memory.buffer_window_chat_memory import BufferWindowChatMemoryConfig
from tinygent.runtime.global_registry import GlobalRegistry


def _register_memories() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_memory(
        'buffer', BufferChatMemoryConfig, BufferChatMemory
    )
    registry.register_memory(
        'buffer_window', BufferWindowChatMemoryConfig, BufferWindowChatMemory
    )


_register_memories()
