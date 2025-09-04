from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinyPlanMessage
from tinygent.memory.base_chat_memory import BaseChatMemory


def main():
    memory = BaseChatMemory()

    print('=== Initial memory ===')
    print(memory.load_variables())
    print()

    # First exchange
    msg1 = TinyHumanMessage(content='Hello, assistant.')
    memory.save_context(msg1)

    msg2 = TinyChatMessage(content='Hi there! How can I help you today?')
    memory.save_context(msg2)

    print('=== After first exchange ===')
    print(memory.load_variables())
    print()

    # Second exchange
    msg3 = TinyHumanMessage(content='Can you make a plan for my weekend?')
    memory.save_context(msg3)

    msg4 = TinyPlanMessage(content='Sure! 1. Go hiking. 2. Watch a movie. 3. Relax.')
    memory.save_context(msg4)

    print('=== After second exchange ===')
    print(memory.load_variables())
    print()

    # Third exchange
    msg5 = TinyHumanMessage(content='That sounds nice, thanks.')
    memory.save_context(msg5)

    msg6 = TinyChatMessage(
        content='You’re welcome! Let me know if you need anything else.'
    )
    memory.save_context(msg6)

    print('=== After third exchange ===')
    print(memory.load_variables())
    print()

    # Clear memory
    memory.clear()
    print('=== After clear() ===')
    print(memory.load_variables())


if __name__ == '__main__':
    main()
