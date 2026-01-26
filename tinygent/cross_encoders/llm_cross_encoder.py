from __future__ import annotations

import logging
from typing import Iterable
from typing import Literal

from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoder
from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoderConfig
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.factory.llm import build_llm
from tinygent.core.prompts.cross_encoders.factory.llm_cross_encoder import (
    get_prompt_template,
)
from tinygent.core.prompts.cross_encoders.template.llm_cross_encoder import (
    LLMCrossEncoderPromptTemplate,
)
from tinygent.core.runtime.executors import run_in_semaphore
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.utils import set_cross_encoder_telemetry_attributes
from tinygent.core.types.base import TinyModel
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = get_prompt_template()


def _validate_score_range(score_range: tuple[float, float]) -> tuple[float, float]:
    """Validate score range so that always two values, and first is min, second is max value of the range."""
    if len(score_range) != 2:
        raise ValueError(
            f'Score range must contain only 2 values (min_value, max_value), got: {score_range}'
        )

    if score_range[0] == score_range[1]:
        raise ValueError(
            f'Score range must have different min and max values, got min == max == {score_range[0]}'
        )

    if score_range[0] > score_range[1]:
        orig = score_range
        score_range = (score_range[1], score_range[0])
        logger.warning(
            'Score range min is greater than max, swapping values: %s -> %s',
            orig,
            score_range,
        )
    return score_range


class LLMCrossEncoderConfig(AbstractCrossEncoderConfig['LLMCrossEncoder']):
    type: Literal['llm'] = Field(default='llm', frozen=True)

    prompt_template: LLMCrossEncoderPromptTemplate = Field(default=_DEFAULT_PROMPT)

    llm: AbstractLLMConfig | AbstractLLM = Field(...)

    score_range: tuple[float, float] = Field(default=(-5.0, 5.0))

    @model_validator(mode='after')
    def validate_(self) -> Self:
        self.score_range = _validate_score_range(self.score_range)
        return self

    def build(self) -> LLMCrossEncoder:
        return LLMCrossEncoder(
            llm=self.llm if isinstance(self.llm, AbstractLLM) else build_llm(self.llm),
            prompt_template=self.prompt_template,
            score_range=self.score_range,
        )


class LLMCrossEncoder(AbstractCrossEncoder):
    """LLM-based cross encoder for ranking text relevance.

    Uses a language model to score the relevance between query-text pairs or
    arbitrary text pairs. Unlike embedding-based approaches, this cross encoder
    evaluates each pair directly through the LLM, potentially capturing more
    nuanced semantic relationships.

    The cross encoder can be used for:
    - Reranking search results based on query relevance
    - Evaluating semantic similarity between text pairs
    - Filtering candidates in information retrieval pipelines

    Scores are generated within a configurable range (default: -5.0 to 5.0),
    where higher scores indicate greater relevance or similarity.

    Args:
        llm: Language model for scoring pairs
        prompt_template: Template for ranking prompts (default provided)
        score_range: (min, max) range for relevance scores (default: (-5.0, 5.0))
    """

    def __init__(
        self,
        llm: AbstractLLM,
        prompt_template: LLMCrossEncoderPromptTemplate = _DEFAULT_PROMPT,
        score_range: tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.score_range = _validate_score_range(score_range)

    @property
    def config(self) -> LLMCrossEncoderConfig:
        return LLMCrossEncoderConfig(
            llm=self.llm.config,
            prompt_template=self.prompt_template,
            score_range=self.score_range,
        )

    async def _single_rank(self, query: str, text: str) -> tuple[tuple[str, str], float]:
        class CrossEncoderResult(TinyModel):
            score: float

        result = await self.llm.agenerate_structured(
            llm_input=TinyLLMInput(
                messages=[
                    TinySystemMessage(content=self.prompt_template.ranking.system),
                    TinyHumanMessage(
                        content=render_template(
                            self.prompt_template.ranking.user,
                            {
                                'query': query,
                                'text': text,
                                'min_range_val': self.score_range[0],
                                'max_range_val': self.score_range[1],
                            },
                        )
                    ),
                ]
            ),
            output_schema=CrossEncoderResult,
        )
        return ((query, text), result.score)

    @tiny_trace('rank')
    async def rank(
        self, query: str, texts: Iterable[str]
    ) -> list[tuple[tuple[str, str], float]]:
        texts_list = list(texts)
        tasks = [self._single_rank(query, text) for text in texts_list]
        result = await run_in_semaphore(*tasks)

        set_cross_encoder_telemetry_attributes(
            self.config,
            query=query,
            texts=texts_list,
            result=result,
        )

        return result

    @tiny_trace('predict')
    async def predict(
        self, pairs: Iterable[tuple[str, str]]
    ) -> list[tuple[tuple[str, str], float]]:
        pairs_list = list(pairs)
        tasks = [self._single_rank(p[0], p[1]) for p in pairs_list]
        result = await run_in_semaphore(*tasks)

        set_cross_encoder_telemetry_attributes(
            self.config,
            pairs=pairs_list,
            result=result,
        )

        return result
