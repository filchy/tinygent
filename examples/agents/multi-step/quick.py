from pathlib import Path

from tinygent.cli.builder import build_agent
from tinygent.cli.utils import discover_and_register_components
from tinygent.logging import setup_logger
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')


def main():
    discover_and_register_components()

    agent = build_agent(tiny_yaml_load(str(Path(__file__).parent / 'agent.yaml')))

    result = agent.run('What is the weather like in Paris?')

    logger.info(f'[RESULT] {result}')
    logger.info(f'[AGENT] {str(agent)}')


if __name__ == '__main__':
    main()
