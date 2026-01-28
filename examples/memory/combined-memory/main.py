from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinyPlanMessage
from tinygent.memory import BufferChatMemory
from tinygent.memory import BufferWindowChatMemory
from tinygent.memory import CombinedMemory


def build_memory() -> CombinedMemory:
    """Create a CombinedMemory with a full buffer and a 3-message window.

    Order matters: later memories override key collisions in load_variables().
    """
    return CombinedMemory(
        memory_list=[
            BufferChatMemory(),
            BufferWindowChatMemory(k=3),
        ]
    )


def populate(memory: CombinedMemory) -> None:
    # First exchange
    memory.save_context(TinyHumanMessage(content='Hello, assistant.'))
    memory.save_context(TinyChatMessage(content='Hi there! How can I help you today?'))

    # Second exchange
    memory.save_context(TinyHumanMessage(content='Can you make a plan for my weekend?'))
    memory.save_context(
        TinyPlanMessage(content='Sure! 1. Go hiking. 2. Watch a movie. 3. Relax.')
    )

    # Third exchange
    memory.save_context(TinyHumanMessage(content='That sounds nice, thanks.'))
    memory.save_context(
        TinyChatMessage(content="You're welcome! Let me know if you need anything else.")
    )


def main() -> None:
    memory = build_memory()
    populate(memory)

    print('=== Combined Memory Keys ===')
    print(memory.memory_keys)
    print()

    print('=== Merged Variables ===')
    vars_dict = memory.load_variables()
    for k, v in vars_dict.items():
        print(f'{k}:\n{textwrap.indent(v, "    ")}\n')

    print('=== Combined __str__ Representation ===')
    print(memory)
    print()

    # Show underlying window explicitly (second memory is BufferWindowChatMemory)
    window_mem = memory.memory_list[1]
    if isinstance(window_mem, BufferWindowChatMemory):
        print(f'=== Sliding Window (last {window_mem.k} messages) ===')
        for m in window_mem.chat_buffer_window:
            print(m.tiny_str)


if __name__ == '__main__':
    import textwrap

    main()
