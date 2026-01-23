from tinygent.core.runtime.global_registry import GlobalRegistry
from tinygent.cross_encoders.llm_cross_encoder import LLMCrossEncoder
from tinygent.cross_encoders.llm_cross_encoder import LLMCrossEncoderConfig


def _register_crossencoders() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_crossencoder('llm', LLMCrossEncoderConfig, LLMCrossEncoder)


_register_crossencoders()
