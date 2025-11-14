from tinygent.datamodels.messages import AllTinyMessages
from tinygent.types import TinyModel


class TinyLLMInput(TinyModel):
    """Input to an LLM, consisting of a list of messages."""

    messages: list[AllTinyMessages]
