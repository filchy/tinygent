from __future__ import annotations

import logging
from typing import Iterable
from typing import Literal

from pydantic import model_validator
from typing_extensions import Self

from tinygent.runtime.executors import run_in_semaphore
from tinygent.datamodels.cross_encoder import AbstractCrossEncoder
from tinygent.datamodels.cross_encoder import AbstractCrossEncoderConfig
from tinygent.telemetry.decorators import tiny_trace
from tinygent.telemetry.utils import set_cross_encoder_telemetry_attributes
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinySystemMessage
from tinygent.factory.llm import build_llm
from tinygent.types.base import TinyModel
from tinygent.types.io.llm_io_input import TinyLLMInput
from tinygent.types.prompt_template import TinyPromptTemplate
from tinygent.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)


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


class LLMCrossEncoderPromptTemplate(TinyPromptTemplate):
    """Prompt template for LLM Cross-encoder."""

    ranking: TinyPromptTemplate.UserSystem

    _template_fields = {
        'ranking.user': {'query', 'text', 'min_range_val', 'max_range_val'}
    }


class LLMCrossEncoderConfig(AbstractCrossEncoderConfig['LLMCrossEncoder']):
    type: Literal['llm'] = 'llm'

    prompt_template: LLMCrossEncoderPromptTemplate | None = None

    llm: AbstractLLMConfig | AbstractLLM

    score_range: tuple[float, float] = (-5.0, 5.0)

    @model_validator(mode='after')
    def validate_(self) -> Self:
        self.score_range = _validate_score_range(self.score_range)
        return self

    def build(self) -> LLMCrossEncoder:
        if not self.prompt_template:
            from ..prompts.cross_encoders.llm_cross_encoder import get_prompt_template

            self.prompt_template = get_prompt_template()

        return LLMCrossEncoder(
            llm=self.llm if isinstance(self.llm, AbstractLLM) else build_llm(self.llm),
            prompt_template=self.prompt_template,
            score_range=self.score_range,
        )


class LLMCrossEncoder(AbstractCrossEncoder):
    def __init__(
        self,
        llm: AbstractLLM,
        prompt_template: LLMCrossEncoderPromptTemplate,
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
