from pathlib import Path

from tinygent.cli.builder import build_agent
from tinygent.cli.utils import discover_and_register_components
from tinygent.logging import setup_logger
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')


def main():
    parent_path = Path(__file__).parent

    discover_and_register_components(str(parent_path / 'main.py'))

    agent = build_agent(tiny_yaml_load(str(parent_path / 'agent.yaml')))

    result = agent.run(
        'Will the Albany in Georgia reach a hundred thousand occupants before the one in New York?'
    )

    logger.info(f'[RESULT] {result}')
    logger.info(f'[AGENT] {str(agent)}')


if __name__ == '__main__':
    main()
