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
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory.llm import build_llm
from tinygent.core.prompts.agents.middleware.factory.llm_tool_selector import (
    get_prompt_template,
)
from tinygent.core.prompts.agents.middleware.template.llm_tool_selector import (
    LLMToolSelectorPromptTemplate,
)
from tinygent.core.types.io.llm_io_input import TinyLLMInput

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

    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        if not kwargs.get('tools'):
            return

        selected_tools = []
        tools = cast(list[AbstractTool], kwargs.get('tools', []))
        mapped_tools = {t.info.name: t for t in tools}

        if self.always_include:
            for name in self.always_include:
                if t := mapped_tools.get(name):
                    selected_tools.append(t)

        remaining_space: int | None = None
        if self.max_tools:
            remaining_space = self.max_tools - len(selected_tools)
            if remaining_space <= 0:
                kwargs['tools'] = selected_tools
                return
