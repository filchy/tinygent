from typing import Annotated
import yaml
from pathlib import Path
from typing import Union

from pydantic import Field
from pydantic import TypeAdapter

from tinygent.agents import TinyBaseAgent
from tinygent.agents import TinyBaseAgentConfig
from tinygent.agents import TinyMultiStepAgentConfig

_all_configs = [TinyBaseAgentConfig, TinyMultiStepAgentConfig]


def build_agent(path: str) -> TinyBaseAgent:
    """Builds an agent from a YAML configuration file."""
    p = Path(path)

    TinyAgentConfig = Annotated[
        Union[tuple(_all_configs)], Field(discriminator='agent_type')
    ]

    if not p.exists():
        raise FileNotFoundError(f'Path {path} does not exist.')

    config = yaml.safe_load(p.read_text())

    config_adapter = TypeAdapter(TinyAgentConfig)
    agent_config = config_adapter.validate_python(config)
    return agent_config.build_agent()
