import asyncio

from pydantic import Field

from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.llms import OpenAILLM
from tinygent.tools import reasoning_tool
from tinygent.tools import tool
from tinygent.types.base import TinyModel


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


def basic_generation():
    llm = OpenAILLM()

    result = llm.generate_text(
        llm_input=TinyLLMInput(
            messages=[TinyHumanMessage(content='Tell me a joke about programmers.')]
        )
    )

    for msg in result.tiny_iter():
        print(f'[BASIC TEXT GENERATION] {msg}')


def structured_generation():
    llm = OpenAILLM()

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
    llm = OpenAILLM()

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
    llm = OpenAILLM()

    result = await llm.agenerate_text(
        llm_input=TinyLLMInput(
            messages=[TinyHumanMessage(content='Name three uses of AI in medicine.')]
        )
    )

    for msg in result.tiny_iter():
        print(f'[ASYNC TEXT GENERATION] {msg}')


async def text_streaming():
    llm = OpenAILLM()

    async for chunk in llm.stream_text(
        llm_input=TinyLLMInput(
            messages=[TinyHumanMessage(content='Tell me a joke about programmers.')]
        )
    ):
        if chunk.is_message:
            assert chunk.message is not None
            print(f'[STREAMED CHUNK] {chunk.message.content}')


async def tool_call_streaming():
    llm = OpenAILLM()

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
    basic_generation()
    structured_generation()
    generation_with_tools()

    asyncio.run(async_generation())
    asyncio.run(text_streaming())
    asyncio.run(tool_call_streaming())
