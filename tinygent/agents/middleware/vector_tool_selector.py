from __future__ import annotations

import logging
from typing import Any
from typing import Callable
from typing import Literal

import numpy as np
from pydantic import Field

from tinygent.agents.middleware.base_tool_selector import TinyBaseToolSelectorMiddleware
from tinygent.agents.middleware.base_tool_selector import (
    TinyBaseToolSelectorMiddlewareConfig,
)
from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.embedder import AbstractEmbedderConfig
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.factory.embedder import build_embedder
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attributes
from tinygent.core.types.io.llm_io_input import TinyLLMInput

logger = logging.getLogger(__name__)


class TinyVectorToolSelectorMiddlewareConfig(
    TinyBaseToolSelectorMiddlewareConfig['TinyVectorToolSelectorMiddleware']
):
    """Configuration for TinyVectorToolSelector Middleware."""

    type: Literal['vector_tool_classifier'] = Field(
        default='vector_tool_classifier', frozen=True
    )

    embedder: AbstractEmbedder | AbstractEmbedderConfig = Field(...)

    similarity_threshold: float | None = Field(default=None)

    query_transform_fn: Callable[[TinyLLMInput], str] | None = Field(default=None)

    tool_transform_fn: Callable[[AbstractTool], str] | None = Field(default=None)

    def build(self) -> TinyVectorToolSelectorMiddleware:
        return TinyVectorToolSelectorMiddleware(
            embedder=self.embedder
            if isinstance(self.embedder, AbstractEmbedder)
            else build_embedder(self.embedder),
            similarity_threshold=self.similarity_threshold,
            query_transform_fn=self.query_transform_fn,
            tool_transform_fn=self.tool_transform_fn,
            **self.build_base_kwargs(),
        )


class TinyVectorToolSelectorMiddleware(TinyBaseToolSelectorMiddleware):
    def __init__(
        self,
        *,
        embedder: AbstractEmbedder,
        max_tools: int | None = None,
        always_include: list[AbstractTool] | None = None,
        similarity_threshold: float | None = None,
        query_transform_fn: Callable[[TinyLLMInput], str] | None = None,
        tool_transform_fn: Callable[[AbstractTool], str] | None = None,
    ):
        super().__init__(max_tools, always_include)

        if not query_transform_fn:

            def _default_query_transform_fn(llm_input: TinyLLMInput) -> str:
                for m in reversed(llm_input.messages):
                    if isinstance(m, TinyHumanMessage):
                        return m.content
                raise ValueError('No human message for embedding found in history')

            query_transform_fn = _default_query_transform_fn

        if not tool_transform_fn:

            def _default_tool_transform_fn(tool: AbstractTool) -> str:
                return f'{tool.info.name} - {tool.info.description}'

            tool_transform_fn = _default_tool_transform_fn

        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.query_transform_fn = query_transform_fn
        self.tool_transform_fn = tool_transform_fn

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_np = np.array(a, dtype=float)
        b_np = np.array(b, dtype=float)

        if a_np.shape != b_np.shape:
            raise ValueError(
                f'Vectors for cosine similarity must have the same shape a[{a_np.shape}], b[{b_np.shape}]'
            )

        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a_np, b_np) / (norm_a * norm_b))

    @tiny_trace('vector_tool_selector.before_llm_call')
    async def before_llm_call(
        self, *, run_id: str, llm_input: TinyLLMInput, kwargs: dict[str, Any]
    ) -> None:
        candidates = self._prepare_candidates(kwargs)

        if candidates is None:
            return

        tools = candidates.tools
        remaining_tools = candidates.remaining_tools
        selected_tools = candidates.selected_tools

        set_tiny_attributes(
            {
                'vector_tool_selector.available_tools': [t.info.name for t in tools],
                'vector_tool_selector.available_tools.total': len(tools),
                'vector_tool_selector.max_tools': str(self.max_tools),
                'vector_tool_selector.always_include': str(self.always_include),
                'vector_tool_selector.remaining_space': str(candidates.remaining_space),
            }
        )

        transformed_query = self.query_transform_fn(llm_input)
        transformed_tools = [self.tool_transform_fn(t) for t in remaining_tools]

        query_emb, *tool_embs = await self.embedder.aembed_batch(
            [transformed_query] + transformed_tools
        )

        similarities = [
            self._cosine_similarity(query_emb, tool_emb) for tool_emb in tool_embs
        ]
        tool_sim_pairs = [
            (s, t, tt)
            for s, t, tt in zip(
                similarities, remaining_tools, transformed_tools, strict=True
            )
        ]
        tool_sim_pairs = sorted(
            tool_sim_pairs, key=lambda x: x[0], reverse=True
        )  # sort by highest similarity

        for s, t, tt in tool_sim_pairs:
            set_tiny_attributes(
                {
                    f'vector_tool_selector.{t.info.name}.transformed_description': tt,
                    f'vector_tool_selector.{t.info.name}.similarity_score': s,
                }
            )

            if self.max_tools and len(selected_tools) >= self.max_tools:
                continue

            if self.similarity_threshold is None or (
                self.similarity_threshold and s > self.similarity_threshold
            ):
                selected_tools.add(t)

        set_tiny_attributes(
            {
                'vector_tool_selector.transformed_query': transformed_query,
                'vector_tool_selector.selected_tools': [
                    t.info.name for t in selected_tools
                ],
                'vector_tool_selector.selected_tools.total': len(selected_tools),
            }
        )

        kwargs['tools'] = selected_tools
