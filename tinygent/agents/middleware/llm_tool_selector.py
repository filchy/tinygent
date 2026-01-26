from __future__ import annotations

import logging
from typing import Any
from typing import Literal
from typing import cast

from pydantic import Field

from tinygent.agents.middleware.base import TinyBaseMiddleware
from tinygent.agents.middleware.base import TinyBaseMiddlewareConfig
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.datamodels.messages import TinyHumanMessage, TinySystemMessage
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory.llm import build_llm
from tinygent.core.prompts.agents.middleware.factory.llm_tool_selector import (
    get_prompt_template,
)
from tinygent.core.prompts.agents.middleware.template.llm_tool_selector import (
    LLMToolSelectorPromptTemplate,
)
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attribute, set_tiny_attributes
from tinygent.core.types.base import TinyModel
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = get_prompt_template()


class TinyLLMToolSelectorMiddlewareConfig(
    TinyBaseMiddlewareConfig['TinyLLMToolSelectorMiddleware']
):
    """Configuration for LLMToolSelector Middleware."""

    type: Literal['llm_tool_selector'] = Field(default='llm_tool_selector', frozen=True)

    llm: AbstractLLMConfig | AbstractLLM = Field(...)

    prompt_template: LLMToolSelectorPromptTemplate = Field(default=_DEFAULT_PROMPT)

    max_tools: int | None = Field(default=None)

    always_include: list[str] | None = Field(default=None)

    def build(self) -> TinyLLMToolSelectorMiddleware:
        return TinyLLMToolSelectorMiddleware(
            llm=self.llm if isinstance(self.llm, AbstractLLM) else build_llm(self.llm),
            prompt_template=self.prompt_template,
            max_tools=self.max_tools,
            always_include=self.always_include,
        )


class TinyLLMToolSelectorMiddleware(TinyBaseMiddleware):
    def __init__(
        self,
        *,
        llm: AbstractLLM,
        prompt_template: LLMToolSelectorPromptTemplate = _DEFAULT_PROMPT,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_tools = max_tools
        self.always_include = always_include

        if always_include and max_tools and len(always_include) > max_tools:
            logger.warning(
                'always_include contains %d items which exceeds max_tools=%d; '
                'increasing max_tools to %d to fit always_include.',
                len(always_include),
                max_tools,
                len(always_include),
            )
            self.max_tools = len(always_include)

    @staticmethod
    def _create_selection_model(tools: list[AbstractTool]) -> type[TinyModel]:
        tool_names = tuple(t.info.name for t in tools)
        tools_literals = Literal[*tool_names]  # type: ignore

        class LocalSelectionModel(TinyModel):
            selected_tools: list[tools_literals]  # type: ignore

        LocalSelectionModel.__name__ = 'ToolSelectionResult'
        LocalSelectionModel.model_rebuild()

        return LocalSelectionModel

    @tiny_trace('tool_selector')
    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        if not kwargs.get('tools'):
            return

        selected_tools = set()
        tools = cast(list[AbstractTool], kwargs.get('tools', []))
        mapped_tools = {t.info.name: t for t in tools}

        set_tiny_attributes(
            {
                'tool_selector.available_tools': [t.info.name for t in tools],
                'tool_selector.available_tools.total': len(tools),
                'tool_selector.max_tools': str(self.max_tools),
                'tool_selector.always_include': str(self.always_include),
            }
        )

        if self.always_include:
            for name in self.always_include:
                if t := mapped_tools.get(name):
                    selected_tools.add(t)

        remaining_space: int | None = None
        if self.max_tools:
            remaining_space = self.max_tools - len(selected_tools)
            if remaining_space <= 0:
                kwargs['tools'] = selected_tools
                return

        local_llm_input = llm_input.model_copy()
        local_llm_input.add_at_end(
            TinySystemMessage(content=self.prompt_template.system)
        )
        local_llm_input.add_at_end(
            TinyHumanMessage(content=render_template(
                self.prompt_template.user,
                {
                    'tools': '\n'.join(
                        [f'{t.info.name} - {t.info.description or 'Description not provided'}' for t in tools]
                    )
                }
            ))
        )

        result = await self.llm.agenerate_structured(
            llm_input=local_llm_input,
            output_schema=self._create_selection_model(tools)
        )

        for selected_tool_name in result.selected_tools:  # type: ignore
            tool_obj = mapped_tools.get(selected_tool_name)

            if tool_obj in selected_tools:
                logger.warning('Tool \'%s\' already selected', selected_tool_name)
                continue

            if not tool_obj:
                logger.error(
                    'Selected tool \'%s\' doesn\'t exist amongs available tools: %s',
                    selected_tool_name,
                    [t.info.name for t in tools]
                )
                continue

            selected_tools.add(tool_obj)

        set_tiny_attributes(
            {
                'tool_selector.selected_tools': [t.info.name for t in selected_tools],
                'tool_selector.selected_tools.total': len(selected_tools),
            }
        )

        kwargs['tools'] = selected_tools
