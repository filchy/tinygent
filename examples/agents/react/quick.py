from pathlib import Path

from tinygent.cli.builder import build_agent
from tinygent.logging import setup_general_loggers
from tinygent.logging import setup_logger
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')
setup_general_loggers('warning')


def main():
    agent = build_agent(tiny_yaml_load(str(Path(__file__).parent / 'agent.yaml')))

    result = agent.run('What is the weather like in Paris?')

    logger.info(f'[RESULT] {result}')


if __name__ == '__main__':
    main()
