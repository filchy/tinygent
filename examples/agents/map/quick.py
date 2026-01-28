from pathlib import Path

from tinygent.cli.utils import discover_and_register_components
from tinygent.core.factory import build_agent
from tinygent.logging import setup_logger
from tinygent.utils import tiny_yaml_load

logger = setup_logger('debug')


def main():
    parent_path = Path(__file__).parent

    # Discover and register tools and middleware from main.py
    discover_and_register_components(str(parent_path / 'main.py'))

    agent = build_agent(tiny_yaml_load(str(parent_path / 'agent.yaml')))

    result = agent.run(
        'Will the Albany in Georgia reach a hundred thousand occupants before the one in New York?'
    )

    logger.info('[RESULT] %s', result)
    logger.info('[AGENT] %s', str(agent))


if __name__ == '__main__':
    main()
