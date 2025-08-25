from tinygent.runtime.global_registry import GlobalRegistry
from tinygent.tools.tool import tool


@tool
def get_weather(location: str) -> str:

    return f'The weather in {location} is sunny with a high of 75°F.'


@tool
def get_time(location: str) -> str:

    return f'The current time in {location} is 2:00 PM.'


if __name__ == '__main__':
    print(f'{GlobalRegistry.get_registry()._registered_tools =}')
