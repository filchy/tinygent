from typing import TypedDict

from google.genai.types import ContentOrDict
from google.genai.types import Part


class GeminiParams(TypedDict):
    message: list[Part]
    history: list[ContentOrDict]
