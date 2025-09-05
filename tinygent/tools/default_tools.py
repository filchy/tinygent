from pydantic import Field

from tinygent.datamodels.messages import TinyChatMessage
from tinygent.tools.tool import tool
from tinygent.types import TinyModel


class TinyFinalAnswerInput(TinyModel):
    response: str = Field(..., description='The final answer to return to the user.')


@tool
def provide_final_answer(data: TinyFinalAnswerInput) -> TinyChatMessage:
    return TinyChatMessage(content=data.response, metadata={'is_final_answer': True})
