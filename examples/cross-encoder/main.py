import asyncio

from tinygent.core.factory import build_cross_encoder


def _print_results(results: list[tuple[tuple[str, str], float]], title: str):
    """Pretty print cross-encoder results."""
    print(f'\n{title}')
    print('=' * 80)
    for (query, text), score in sorted(results, key=lambda x: x[1], reverse=True):
        print(f'Score: {score:6.2f} | Query: "{query[:30]}..." | Text: "{text[:50]}..."')
    print()


async def single_ranking():
    """Demonstrate ranking multiple texts against a single query."""
    cross_encoder = build_cross_encoder('voyageai:rerank-2.5')

    query = 'How to build AI agents?'
    texts = [
        'TinyGent is a lightweight framework for building AI agents with tools and memory.',
        'The weather today is sunny with a high of 75°F.',
        'Python is a high-level programming language.',
        'ReAct agents combine reasoning and acting to solve complex tasks step by step.',
        'Chocolate cake recipe requires flour, sugar, eggs, and cocoa powder.',
    ]

    results = await cross_encoder.rank(query=query, texts=texts)
    _print_results(results, '[SINGLE RANKING] Ranking texts by relevance to query')


async def batch_prediction():
    """Demonstrate scoring multiple query-text pairs."""
    cross_encoder = build_cross_encoder('llm', llm='openai:gpt-4o-mini')

    pairs = [
        ('What is TinyGent?', 'TinyGent is a framework for building AI agents.'),
        ('What is TinyGent?', 'The capital of France is Paris.'),
        ('How to cook pasta?', 'Boil water, add pasta, cook for 8-10 minutes.'),
        ('How to cook pasta?', 'Machine learning models require training data.'),
        ('Best travel destinations', 'Paris, Tokyo, and Barcelona are popular cities.'),
        (
            'Best travel destinations',
            'Integer overflow can cause security vulnerabilities.',
        ),
    ]

    results = await cross_encoder.predict(pairs=pairs)
    _print_results(results, '[BATCH PREDICTION] Scoring query-text pairs')


async def compare_agents_relevance():
    """Demonstrate using cross-encoder to rank agent descriptions by relevance."""
    cross_encoder = build_cross_encoder('llm', llm='openai:gpt-4o-mini')

    query = (
        'I need an agent that can use tools and reason about complex multi-step problems'
    )

    agent_descriptions = [
        'ReAct Agent: Combines reasoning and acting in iterative loops to solve tasks using tools.',
        'Multi-Step Agent: Breaks down complex tasks into sequential steps with tool usage.',
        'Map Agent: Applies the same operation across multiple items in parallel.',
        'Squad Agent: Coordinates multiple specialized agents to work together on tasks.',
        'Simple Chat Agent: Provides basic conversational responses without tool usage.',
    ]

    results = await cross_encoder.rank(query=query, texts=agent_descriptions)

    print('\n[AGENT SELECTION] Finding the most relevant agent for the task')
    print('=' * 80)
    print(f'Query: "{query}"\n')
    for i, ((q, text), score) in enumerate(
        sorted(results, key=lambda x: x[1], reverse=True), 1
    ):
        agent_name = text.split(':')[0]
        description = text.split(':', 1)[1].strip()
        print(f'{i}. {agent_name} (score: {score:.2f})')
        print(f'   {description}\n')


async def custom_score_range():
    """Demonstrate using a custom score range."""
    # Using 0-100 range instead of default -5 to 5
    cross_encoder = build_cross_encoder(
        'llm', llm='openai:gpt-4o-mini', score_range=(0, 100)
    )

    query = 'Python programming tutorials'
    texts = [
        'Learn Python from scratch with interactive examples and exercises.',
        'JavaScript is the language of the web for frontend development.',
        'Advanced Python techniques for data science and machine learning.',
    ]

    results = await cross_encoder.rank(query=query, texts=texts)

    print('\n[CUSTOM SCORE RANGE] Using 0-100 scoring scale')
    print('=' * 80)
    for (query, text), score in sorted(results, key=lambda x: x[1], reverse=True):
        print(f'Score: {score:5.1f}/100 | {text}')
    print()


async def main():
    """Run all cross-encoder examples."""
    await single_ranking()
    await batch_prediction()
    await compare_agents_relevance()
    await custom_score_range()


if __name__ == '__main__':
    asyncio.run(main())
