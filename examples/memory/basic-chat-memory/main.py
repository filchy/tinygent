from tinygent.datamodels.llm_io import TinyLLMInput
from tinygent.llms.openai import OpenAILLM
from tinygent.memory.base_chat_memory import BaseChatMemory


def main():
    memory = BaseChatMemory()
    llm = OpenAILLM()

    print('=== Initial memory ===')
    print(memory.load_variables())
    print()

    # 1) First exchange
    prompt1 = TinyLLMInput(text='Say hello in one short sentence.')
    result1 = llm.generate_text(prompt1)
    memory.save_context(prompt1, result1)

    print('=== After first exchange ===')
    print(memory.load_variables())
    print()

    # 2) Second exchange (model sees prior context if you pass memory vars)
    # In a real chain, you'd merge memory.load_variables() into your prompt.
    prompt2 = TinyLLMInput(text='What did you just greet me with? Answer briefly.')
    result2 = llm.generate_text(prompt2)
    memory.save_context(prompt2, result2)

    print('=== After second exchange ===')
    print(memory.load_variables())
    print()

    # 3) Clear memory
    memory.clear()
    print('=== After clear() ===')
    print(memory.load_variables())


if __name__ == '__main__':
    main()
