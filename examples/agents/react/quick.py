from pathlib import Path

from tinygent.cli.utils import discover_and_register_components
from tinygent.factory import build_agent
from tinygent.logging import setup_logger

logger = setup_logger('debug')


def main():
    parent_path = Path(__file__).parent

    discover_and_register_components(str(parent_path / 'main.py'))

    agent = build_agent(
        'react',
        llm='openai:gpt-4o-mini',
        tools=['get_best_destination'],
        memory='buffer',
    )

    result = agent.run(
        'What is the best travel destination and what is the weather like there?'
    )

    logger.info('[RESULT] %s', result)
    logger.info('[AGENT SUMMARY] %s', str(agent))


if __name__ == '__main__':
    main()
