from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinyPlanMessage
from tinygent.core.factory import build_llm
from tinygent.memory import BufferSummaryChatMemory


def main():
    memory = BufferSummaryChatMemory(
        build_llm('openai:gpt-4o-mini'),
        max_token_limit=30,
        return_messages=True,
    )

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

    print('=== Full memory ===')
    print(memory.load_variables())
    print()


if __name__ == '__main__':
    main()
