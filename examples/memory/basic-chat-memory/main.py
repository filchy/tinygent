from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinyPlanMessage
from tinygent.memory import BufferChatMemory


def main():
    memory = BufferChatMemory()

    # First exchange
    msg1 = TinyHumanMessage(content='Hello, assistant.')
    memory.save_context(msg1)

    msg2 = TinyChatMessage(content='Hi there! How can I help you today?')
    memory.save_context(msg2)

    # Second exchange
    msg3 = TinyHumanMessage(content='Can you make a plan for my weekend?')
    memory.save_context(msg3)

    msg4 = TinyPlanMessage(content='Sure! 1. Go hiking. 2. Watch a movie. 3. Relax.')
    memory.save_context(msg4)

    # Third exchange
    msg5 = TinyHumanMessage(content='That sounds nice, thanks.')
    memory.save_context(msg5)

    msg6 = TinyChatMessage(
        content='You’re welcome! Let me know if you need anything else.'
    )
    memory.save_context(msg6)

    print('=== Full memory ===')
    print(memory._chat_history)
    print()

    # Add filter: only human messages
    memory._chat_history.add_filter(
        'only_human', lambda m: isinstance(m, TinyHumanMessage)
    )
    print('=== Filtered memory (only human) ===')
    print(memory._chat_history)
    print()

    # Replace with filter: only AI/chat messages
    memory._chat_history.remove_filter('only_human')
    memory._chat_history.add_filter(
        'only_ai', lambda m: isinstance(m, (TinyChatMessage, TinyPlanMessage))
    )
    print('=== Filtered memory (only AI/plan) ===')
    print(memory._chat_history)
    print()

    # Remove all filters
    memory._chat_history.remove_filter('only_ai')
    print('=== Memory again (all messages) ===')
    print(memory._chat_history)


if __name__ == '__main__':
    main()
