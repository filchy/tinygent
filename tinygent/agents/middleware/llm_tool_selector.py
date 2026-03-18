from __future__ import annotations

import logging
from typing import Any
from typing import Literal

from pydantic import Field

from tinygent.agents.middleware.base_tool_selector import TinyBaseToolSelectorMiddleware
from tinygent.agents.middleware.base_tool_selector import (
    TinyBaseToolSelectorMiddlewareConfig,
)
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory.llm import build_llm
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attributes
from tinygent.core.types.base import TinyModel
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.prompts.middleware import LLMToolSelectorPromptTemplate
from tinygent.prompts.middleware import get_llm_tool_selector_prompt_template
from tinygent.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)


class TinyLLMToolSelectorMiddlewareConfig(
    TinyBaseToolSelectorMiddlewareConfig['TinyLLMToolSelectorMiddleware']
):
    """Configuration for LLMToolSelector Middleware."""

    type: Literal['llm_tool_selector'] = Field(default='llm_tool_selector', frozen=True)

    llm: AbstractLLMConfig | AbstractLLM = Field(...)

    prompt_template: LLMToolSelectorPromptTemplate = Field(
        default_factory=get_llm_tool_selector_prompt_template
    )

    def build(self) -> TinyLLMToolSelectorMiddleware:
        return TinyLLMToolSelectorMiddleware(
            llm=self.llm if isinstance(self.llm, AbstractLLM) else build_llm(self.llm),
            prompt_template=self.prompt_template,
            **self.build_base_kwargs(),
        )


class TinyLLMToolSelectorMiddleware(TinyBaseToolSelectorMiddleware):
    """Middleware that intelligently selects relevant tools using an LLM.

    Before each LLM call, this middleware uses a dedicated selection LLM to analyze
    the conversation context and determine which tools are most relevant. This reduces
    token usage and improves performance by only providing the main agent with the
    most appropriate subset of tools.

    The middleware can limit the number of selected tools and always include critical
    tools regardless of the selection process.

    Args:
        llm: LLM to use for tool selection (typically a fast, cost-effective model)
        prompt_template: Template for tool selection prompt (default provided)
        max_tools: Maximum number of tools to select (None = no limit)
        always_include: List of tools to always include in selection (None = no always-include list)
    """

    def __init__(
        self,
        *,
        llm: AbstractLLM,
        prompt_template: LLMToolSelectorPromptTemplate | None = None,
        max_tools: int | None = None,
        always_include: list[AbstractTool] | None = None,
    ) -> None:
        super().__init__(max_tools, always_include)

        self.llm = llm
        self.prompt_template = prompt_template or get_llm_tool_selector_prompt_template()

    @staticmethod
    def _create_selection_model(tools: list[AbstractTool]) -> type[TinyModel]:
        tool_names = tuple(t.info.name for t in tools)
        tools_literals = Literal[*tool_names]  # type: ignore

        class LocalSelectionModel(TinyModel):
            selected_tools: list[tools_literals]  # type: ignore

        LocalSelectionModel.__name__ = 'ToolSelectionResult'
        LocalSelectionModel.model_rebuild()

        return LocalSelectionModel

    @tiny_trace('llm_tool_selector.before_llm_call')
    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        candidates = self._prepare_candidates(kwargs)

        if candidates is None:
            return

        selected_tools = candidates.selected_tools
        tools = candidates.tools
        remaining_tools = candidates.remaining_tools
        mapped_tools = candidates.mapped_tools

        set_tiny_attributes(
            {
                'llm_tool_selector.available_tools': [t.info.name for t in tools],
                'llm_tool_selector.available_tools.total': len(tools),
                'llm_tool_selector.max_tools': str(self.max_tools),
                'llm_tool_selector.always_include': str(self.always_include),
                'llm_tool_selector.remaining_space': str(candidates.remaining_space),
            }
        )

        if len(remaining_tools) > 0:
            local_llm_input = llm_input.model_copy()
            local_llm_input.add_at_end(
                TinySystemMessage(content=self.prompt_template.system)
            )
            local_llm_input.add_at_end(
                TinyHumanMessage(
                    content=render_template(
                        self.prompt_template.user,
                        {
                            'tools': '\n'.join(
                                [
                                    f'{t.info.name} - {t.info.description or "Description not provided"}'
                                    for t in remaining_tools
                                ]
                            )
                        },
                    )
                )
            )

            result = await self.llm.agenerate_structured(
                llm_input=local_llm_input,
                output_schema=self._create_selection_model(remaining_tools),
            )

            for selected_tool_name in result.selected_tools:  # type: ignore
                if self.max_tools and len(selected_tools) >= self.max_tools:
                    break

                tool_obj = mapped_tools.get(selected_tool_name)

                if tool_obj in selected_tools:
                    logger.warning("Tool '%s' already selected", selected_tool_name)
                    continue

                if not tool_obj:
                    logger.error(
                        "Selected tool '%s' doesn't exist amongs available tools: %s",
                        selected_tool_name,
                        [t.info.name for t in remaining_tools],
                    )
                    continue

                selected_tools.add(tool_obj)

        set_tiny_attributes(
            {
                'llm_tool_selector.selected_tools': [
                    t.info.name for t in selected_tools
                ],
                'llm_tool_selector.selected_tools.total': len(selected_tools),
            }
        )

        kwargs['tools'] = selected_tools
