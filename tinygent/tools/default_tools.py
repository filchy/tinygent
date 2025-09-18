from pydantic import Field

from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyReasoningMessage
from tinygent.tools.tool import tool
from tinygent.types import TinyModel


class TinyFinalAnswerInput(TinyModel):
    response: str = Field(..., description='The final answer to return to the user.')


@tool(hidden=True)
def provide_final_answer(data: TinyFinalAnswerInput) -> TinyChatMessage:
    """Provide the final answer to the user."""
    return TinyChatMessage(content=data.response, metadata={'is_final_answer': True})


class TinyReasoningInput(TinyModel):
    reasoning: str = Field(..., description='The reasoning step to log.')


@tool(hidden=True)
def log_reasoning_step(data: TinyReasoningInput) -> TinyReasoningMessage:
    """Log the reasoning step."""
    return TinyReasoningMessage(content=data.reasoning)
