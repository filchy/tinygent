from pydantic import Field

from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.factory import build_llm
from tinygent.core.types import TinyLLMInput
from tinygent.core.types import TinyModel
from tinygent.tools import reasoning_tool
from tinygent.tools import tool


class AddInput(TinyModel):
    a: int = Field(..., description='First number')
    b: int = Field(..., description='Second number')


@tool
def add(data: AddInput) -> int:
    return data.a + data.b


class CapitalizeInput(TinyModel):
    text: str = Field(..., description='Text to capitalize')


@reasoning_tool
def capitalize(data: CapitalizeInput) -> str:
    return data.text.upper()


class SummaryResponse(TinyModel):
    summary: str


def count_tokens():
    llm = build_llm('openai:gpt-4o-mini')

    messages = [
        TinySystemMessage(content='You are helpful tiny assistant.'),
        TinyHumanMessage(content='Tell me a joke about programmers.'),
    ]

    result = llm.count_tokens_in_messages(
        messages=messages,
    )

    print(f'[NUMBER OF TOKENS] {result} for {"\n".join([m.tiny_str for m in messages])}')


def basic_generation():
    llm = build_llm('openai:gpt-4o-mini')

    result = llm.generate_text(
        llm_input=TinyLLMInput(
            messages=[TinyHumanMessage(content='Tell me a joke about programmers.')]
        )
    )

    for msg in result.tiny_iter():
        print(f'[BASIC TEXT GENERATION] {msg}')

    print(f'[TEXT GENERATION - to_string()] {result.to_string()}')


def structured_generation():
    llm = build_llm('openai:gpt-4o-mini')

    result = llm.generate_structured(
        llm_input=TinyLLMInput(
            messages=[
                TinyHumanMessage(
                    content='Summarize why the sky is blue in one sentence.'
                )
            ],
        ),
        output_schema=SummaryResponse,
    )

    print(f'[STRUCTURED RESULT] {result.summary}')


def generation_with_tools():
    llm = build_llm('openai:gpt-4o-mini')

    tools_list = [add, capitalize]
    tools = {tool.info.name: tool for tool in tools_list}

    result = llm.generate_with_tools(
        llm_input=TinyLLMInput(
            messages=[
                TinyHumanMessage(
                    content='Capitalize "tinygent is powerful". Then add 5 and 7.'
                )
            ]
        ),
        tools=tools_list,
    )

    for message in result.tiny_iter():
        if message.type == 'chat':
            print(f'[LLM RESPONSE] {message.content}')
        elif message.type == 'tool':
            output = tools[message.tool_name](**message.arguments)
            print(f'[TOOL CALL] {message.tool_name}({message.arguments}) => {output}')


async def async_generation():
    llm = build_llm('openai:gpt-4o-mini')

    result = await llm.agenerate_text(
        llm_input=TinyLLMInput(
            messages=[TinyHumanMessage(content='Name three uses of AI in medicine.')]
        )
    )

    for msg in result.tiny_iter():
        print(f'[ASYNC TEXT GENERATION] {msg}')


async def text_streaming():
    llm = build_llm('openai:gpt-4o-mini')

    async for chunk in llm.stream_text(
        llm_input=TinyLLMInput(
            messages=[TinyHumanMessage(content='Tell me a joke about programmers.')]
        )
    ):
        if chunk.is_message:
            assert chunk.message is not None
            print(f'[STREAMED CHUNK] {chunk.message.content}')


async def tool_call_streaming():
    llm = build_llm('openai:gpt-4o-mini')

    tools = [add, capitalize]

    async for chunk in llm.stream_with_tools(
        llm_input=TinyLLMInput(
            messages=[
                TinyHumanMessage(
                    content='Capitalize "tinygent is powerful". Then add 5 and 7.'
                )
            ]
        ),
        tools=tools,
    ):
        print(f'[STREAMED CHUNK] {chunk}')


if __name__ == '__main__':

    async def main():
        count_tokens()
        basic_generation()
        structured_generation()
        generation_with_tools()

        await async_generation()
        await text_streaming()
        await tool_call_streaming()

    import asyncio

    asyncio.run(main())
