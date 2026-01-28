# Building Agents Guide

A comprehensive guide to building agents with Tinygent.

---

## Quick Start

### 1. Simple Agent

The fastest way to create an agent:

```python
from tinygent.tools import tool
from tinygent.core.factory import build_agent

@tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return f'The weather in {location} is sunny with a high of 75°F.'

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
)

result = agent.run('What is the weather in Prague?')
print(result)
```

---

## Step-by-Step: Building a Travel Agent

Let's build a complete travel planning agent.

### Step 1: Define Tools

```python
from pydantic import Field
from tinygent.core.types import TinyModel
from tinygent.tools import register_tool

class WeatherInput(TinyModel):
    location: str = Field(..., description='City or location name')
    days: int = Field(1, description='Number of days for forecast')

@register_tool
def get_weather(data: WeatherInput) -> str:
    """Get weather forecast for a location."""
    # In real app, call weather API
    return f"{data.days}-day forecast for {data.location}: Sunny and warm"

class FlightInput(TinyModel):
    origin: str = Field(..., description='Departure city')
    destination: str = Field(..., description='Arrival city')
    date: str = Field(..., description='Travel date (YYYY-MM-DD)')

@register_tool
def search_flights(data: FlightInput) -> str:
    """Search for available flights."""
    # In real app, call flight API
    return f"Found 5 flights from {data.origin} to {data.destination} on {data.date}"

class HotelInput(TinyModel):
    location: str = Field(..., description='City name')
    checkin: str = Field(..., description='Check-in date')
    checkout: str = Field(..., description='Check-out date')
    guests: int = Field(1, description='Number of guests')

@register_tool
def search_hotels(data: HotelInput) -> str:
    """Search for available hotels."""
    # In real app, call hotel API
    return f"Found 10 hotels in {data.location} for {data.guests} guest(s)"
```

### Step 2: Create Agent with Memory

```python
from tinygent.core.factory import build_agent
from tinygent.memory import BufferChatMemory

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather, search_flights, search_hotels],
    memory=BufferChatMemory(),
    max_iterations=10,
)
```

### Step 3: Add Middleware for Logging

```python
from tinygent.agents.middleware import TinyBaseMiddleware

class TravelAgentMiddleware(TinyBaseMiddleware):
    def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        print(f"Planning: {reasoning}")

    def before_tool_call(self, *, run_id: str, tool, args) -> None:
        print(f"Searching: {tool.info.name}")

    def on_answer(self, *, run_id: str, answer: str) -> None:
        print(f"Recommendation ready!")

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather, search_flights, search_hotels],
    memory=BufferChatMemory(),
    middleware=[TravelAgentMiddleware()],
)
```

### Step 4: Use the Agent

```python
# First conversation
result = agent.run(
    'I want to travel from New York to Paris in June. '
    'Check the weather and find flights and hotels.'
)
print(result)

# Follow-up conversation (uses memory)
result = agent.run(
    'Actually, make it a 5-day trip and I need accommodations for 2 people.'
)
print(result)
```

---

## Advanced: Multi-Step Planning Agent

For complex workflows, use MultiStep agent:

```python
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.core.factory import build_llm
from tinygent.prompts.multistep import (
    MultiStepPromptTemplate,
    PlanPromptTemplate,
    ActionPromptTemplate,
    FallbackAnswerPromptTemplate,
)

# Custom prompts
plan_prompt = PlanPromptTemplate(
    init_plan=(
        'Create a detailed travel plan for: {{ task }}\n'
        'Available tools: {{ tools }}\n'
        'Break down into clear steps.'
    ),
    update_plan=(
        'Update the travel plan based on new information.\n'
        'Task: {{ task }}\n'
        'Completed steps: {{ steps }}\n'
        'Remaining: {{ remaining_steps }}'
    ),
)

action_prompt = ActionPromptTemplate(
    system='You are a professional travel planning assistant.',
    final_answer=(
        'Provide a comprehensive travel itinerary for: {{ task }}\n'
        'Based on steps completed: {{ steps }}\n'
        'And tool results: {{ tool_calls }}'
    ),
)

fallback_prompt = FallbackAnswerPromptTemplate(
    fallback_answer=(
        'Create final travel plan for: {{ task }}\n'
        'Using information from: {{ history }}'
    )
)

prompt_template = MultiStepPromptTemplate(
    plan=plan_prompt,
    acter=action_prompt,
    fallback=fallback_prompt,
)

# Create agent
agent = TinyMultiStepAgent(
    llm=build_llm('openai:gpt-4o'),
    tools=[get_weather, search_flights, search_hotels],
    prompt_template=prompt_template,
    memory=BufferChatMemory(),
    max_iterations=15,
)

result = agent.run(
    'Plan a 7-day vacation to Tokyo in April for 2 people. '
    'Include flights from San Francisco, hotels, and weather forecast.'
)
```

---

## Multi-Agent System with Squad

Coordinate specialized agents:

```python
from tinygent.agents.squad_agent import TinySquadAgent

# Create specialized agents
weather_agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[get_weather],
)

booking_agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[search_flights, search_hotels],
)

# Create squad coordinator
squad = TinySquadAgent(
    llm=build_llm('openai:gpt-4o'),
    agents=[weather_agent, booking_agent],
    max_iterations=5,
)

result = squad.run(
    'I need a complete travel plan: check weather in Paris, '
    'find flights from London, and book a hotel for 3 nights.'
)
```

---

## Best Practices

### 1. Start Simple, Grow Complex

```python
# Phase 1: Single tool, single agent
agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[weather])

# Phase 2: Multiple tools
agent = build_agent('react', llm='...', tools=[weather, flights, hotels])

# Phase 3: Add memory
agent = build_agent('react', llm='...', tools=[...], memory=BufferChatMemory())

# Phase 4: Add middleware
agent = build_agent('react', llm='...', tools=[...], middleware=[logger])

# Phase 5: Multi-step or Squad
agent = TinyMultiStepAgent(llm='...', tools=[...])
```

### 2. Use Appropriate Agent Type

```python
# Simple Q&A → ReAct
agent = build_agent('react', ...)

# Complex planning → MultiStep
agent = TinyMultiStepAgent(...)

# Specialized tasks → Squad
squad = TinySquadAgent(agents=[specialist1, specialist2])

# Dynamic replanning → MAP
agent = TinyMAPAgent(...)
```

### 3. Handle Errors Gracefully

```python
try:
    result = agent.run('User query')
except Exception as e:
    print(f"Agent error: {e}")
    # Fallback logic
    result = "I'm sorry, I encountered an error. Please try again."
```

### 4. Test with Cheap Models First

```python
# Development
dev_agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[...])

# Production
prod_agent = build_agent('react', llm='openai:gpt-4o', tools=[...])
```

### 5. Monitor and Log

```python
from tinygent.logging import setup_logger

logger = setup_logger('debug')

class LoggingMiddleware(TinyBaseMiddleware):
    def on_reasoning(self, *, run_id: str, reasoning: str) -> None:
        logger.info(f'[{run_id}] Reasoning: {reasoning}')
```

---

## Production Checklist

Before deploying to production:

- [ ] **Error handling**: Wrap agent calls in try-except
- [ ] **Logging**: Add comprehensive logging middleware
- [ ] **Cost tracking**: Monitor LLM API costs
- [ ] **Rate limiting**: Implement rate limits for APIs
- [ ] **Caching**: Cache tool results where appropriate
- [ ] **Memory management**: Clear or summarize long conversations
- [ ] **Monitoring**: Track agent performance and errors
- [ ] **Security**: Validate tool inputs, sanitize outputs
- [ ] **Testing**: Test with edge cases and failure scenarios
- [ ] **Documentation**: Document agent capabilities and limitations

---

## Common Patterns

### Pattern 1: Conversational Agent

```python
from tinygent.memory import BufferChatMemory

agent = build_agent(
    'react',
    llm='openai:gpt-4o-mini',
    tools=[...],
    memory=BufferChatMemory(),
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    response = agent.run(user_input)
    print(f"Agent: {response}")
```

### Pattern 2: Streaming Responses

```python
import asyncio

async def chat_with_streaming():
    agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[...])

    print("Agent: ", end='', flush=True)
    async for chunk in agent.run_stream(user_input):
        print(chunk, end='', flush=True)
    print()  # Newline

asyncio.run(chat_with_streaming())
```

### Pattern 3: Batch Processing

```python
tasks = [
    'What is the weather in Paris?',
    'Find flights from London to Tokyo',
    'Search hotels in New York',
]

agent = build_agent('react', llm='openai:gpt-4o-mini', tools=[...])

results = []
for task in tasks:
    result = agent.run(task)
    results.append(result)
```

---

## Next Steps

- **[Custom Tools Guide](custom-tools.md)**: Build advanced tools
- **[Core Concepts](../concepts/agents.md)**: Deep dive into agents
- **[Examples](../examples.md)**: More examples

---

## Further Reading

- **Agent Architecture**: See `tinygent/agents/` for implementation details
- **Registry Pattern**: See `tinygent/core/runtime/` for global registries
- **Prompts**: See `tinygent/core/prompts/` for prompt templates
