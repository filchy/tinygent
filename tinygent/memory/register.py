from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.memory.buffer_chat_memory import BufferChatMemoryConfig
from tinygent.memory.buffer_window_chat_memory import BufferWindowChatMemory
from tinygent.memory.buffer_window_chat_memory import BufferWindowChatMemoryConfig
from tinygent.memory.combined_memory import CombinedMemory
from tinygent.memory.combined_memory import CombinedMemoryConfig
from tinygent.runtime.global_registry import GlobalRegistry


def _register_memories() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_memory('buffer', BufferChatMemoryConfig, BufferChatMemory)
    registry.register_memory(
        'buffer_window', BufferWindowChatMemoryConfig, BufferWindowChatMemory
    )
    registry.register_memory('combined', CombinedMemoryConfig, CombinedMemory)


_register_memories()
