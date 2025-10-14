from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.memory.buffer_chat_memory import BufferChatMemoryConfig
from tinygent.memory.buffer_window_chat_memory import BufferWindowChatMemory
from tinygent.memory.buffer_window_chat_memory import BufferWindowChatMemoryConfig
from tinygent.runtime.global_registry import GlobalRegistry

GlobalRegistry().get_registry().register_memory(
    'buffer', BufferChatMemoryConfig, BufferChatMemory
)
GlobalRegistry().get_registry().register_memory(
    'buffer_window', BufferWindowChatMemoryConfig, BufferWindowChatMemory
)
