"""
TinyGent terminal command-line interface module.

This subcommand runs the interactive terminal environment for TinyGent.
"""

import logging
from typing import Annotated

from click import Path
import typer

from tinygent.cli.builder import build_agent
from tinygent.utils.yaml import tiny_yaml_load

logger = logging.getLogger(__name__)


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            '--config',
            '-c',
            help='Path to the configuration .yaml/.yml file.',
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
):
    data = tiny_yaml_load(str(config_path))
    agent = build_agent(data)
    for tool in agent._tools:
        logger.info(f'Loaded tool: {tool.info.name} - {tool.info.description}')
