from langchain_core.prompt_values import StringPromptValue
from pydantic import BaseModel
from pydantic import Field

import asyncio

from tinygent.llms.openai import OpenAILLM
from tinygent.runtime.global_registry import GlobalRegistry
from tinygent.tools.tool import tool


class AddInput(BaseModel):
    a: int = Field(..., description='First number')
    b: int = Field(..., description='Second number')


@tool
def add(data: AddInput) -> int:
    return data.a + data.b


class CapitalizeInput(BaseModel):
    text: str = Field(..., description='Text to capitalize')


@tool
def capitalize(data: CapitalizeInput) -> str:
    return data.text.upper()


class SummaryResponse(BaseModel):
    summary: str


def basic_generation():
    llm = OpenAILLM()

    result = llm.generate_text(
        prompt=StringPromptValue(text='Tell me a joke about programmers.')
    )

    for msg in result.tiny_iter():
        print(f'[BASIC TEXT GENERATION] {msg}')


def structured_generation():
    llm = OpenAILLM()

    result = llm.generate_structured(
        prompt=StringPromptValue(text='Summarize why the sky is blue in one sentence.'),
        output_schema=SummaryResponse
    )

    print(f'[STRUCTURED RESULT] {result.summary}')


def generation_with_tools():
    llm = OpenAILLM()

    tools = [add, capitalize]

    result = llm.generate_with_tools(
        prompt=StringPromptValue(
            text='Capitalize 'tinygent is powerful'. Then add 5 and 7.'
        ),
        tools=tools
    )

    for message in result.tiny_iter():
        if message.type == 'chat':
            print(f'[LLM RESPONSE] {message.content}')
        elif message.type == 'tool':
            tool_fn = GlobalRegistry.get_registry().get_tool(message.tool_name)
            output = tool_fn(**message.arguments)
            print(f'[TOOL CALL] {message.tool_name}({message.arguments}) => {output}')


async def async_generation():
    llm = OpenAILLM()

    result = await llm.agenerate_text(
        prompt=StringPromptValue(text='Name three uses of AI in medicine.')
    )

    for msg in result.tiny_iter():
        print(f'[ASYNC TEXT GENERATION] {msg}')


if __name__ == '__main__':
    basic_generation()
    structured_generation()
    generation_with_tools()

    asyncio.run(async_generation())
