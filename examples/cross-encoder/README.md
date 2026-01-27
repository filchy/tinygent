# Cross-Encoder Example

This example demonstrates how to use the **LLM-based Cross-Encoder** (`LLMCrossEncoder`) in **tinygent**. Cross-encoders are powerful tools for semantic ranking and relevance scoring, useful for:

* **Information retrieval** - Ranking search results by relevance
* **Semantic similarity** - Measuring how closely texts match a query
* **Agent selection** - Choosing the best agent for a task based on descriptions
* **Document re-ranking** - Improving search results after initial retrieval

Cross-encoders evaluate query-text pairs jointly, providing more accurate relevance scores than simple embedding similarity.

## Quick Start

```bash
uv sync --extra openai

uv run examples/cross-encoder/main.py
```

---

## Requirements

* You must implement a Cross-Encoder class inheriting from `AbstractCrossEncoder`
* The cross-encoder is created using the `build_cross_encoder()` factory function
* For LLM-based cross-encoders, you need to provide an LLM instance or configuration

---

## Interface: `AbstractCrossEncoder`

The `AbstractCrossEncoder` defines the following required methods:

```python
class AbstractCrossEncoder(ABC):

    async def rank(
        query: str,
        texts: Iterable[str]
    ) -> list[tuple[tuple[str, str], float]]

    async def predict(
        pairs: list[tuple[str, str]]
    ) -> list[tuple[tuple[str, str], float]]
```

Both methods return a list of tuples, where each tuple contains:
- A pair `(query, text)` representing the input
- A float score representing the relevance/similarity

---

## Example 1: Single Query Ranking

Rank multiple texts against a single query to find the most relevant ones:

```python
async def single_ranking():
    cross_encoder = build_cross_encoder('llm', llm='openai:gpt-4o-mini')

    query = 'How to build AI agents?'
    texts = [
        'TinyGent is a lightweight framework for building AI agents with tools and memory.',
        'The weather today is sunny with a high of 75°F.',
        'Python is a high-level programming language.',
        'ReAct agents combine reasoning and acting to solve complex tasks step by step.',
        'Chocolate cake recipe requires flour, sugar, eggs, and cocoa powder.',
    ]

    results = await cross_encoder.rank(query=query, texts=texts)

    # Results are sorted by relevance score (highest first)
    for (query, text), score in sorted(results, key=lambda x: x[1], reverse=True):
        print(f'Score: {score:.2f} | {text}')
```

**Output:**
```
Score: 4.50 | TinyGent is a lightweight framework for building AI agents...
Score: 4.20 | ReAct agents combine reasoning and acting to solve complex tasks...
Score: 1.00 | Python is a high-level programming language.
Score: -3.80 | The weather today is sunny with a high of 75°F.
Score: -4.50 | Chocolate cake recipe requires flour, sugar, eggs...
```

---

## Example 2: Batch Pair Prediction

Score multiple query-text pairs independently:

```python
async def batch_prediction():
    cross_encoder = build_cross_encoder('llm', llm='openai:gpt-4o-mini')

    pairs = [
        ('What is TinyGent?', 'TinyGent is a framework for building AI agents.'),
        ('What is TinyGent?', 'The capital of France is Paris.'),
        ('How to cook pasta?', 'Boil water, add pasta, cook for 8-10 minutes.'),
        ('How to cook pasta?', 'Machine learning models require training data.'),
    ]

    results = await cross_encoder.predict(pairs=pairs)
```

This is useful when you have different queries and want to score each against specific texts.

---

## Example 3: Agent Selection

Use cross-encoders to intelligently select the best agent for a user's task:

```python
async def compare_agents_relevance():
    cross_encoder = build_cross_encoder('llm', llm='openai:gpt-4o-mini')

    query = 'I need an agent that can use tools and reason about complex multi-step problems'

    agent_descriptions = [
        'ReAct Agent: Combines reasoning and acting in iterative loops to solve tasks using tools.',
        'Multi-Step Agent: Breaks down complex tasks into sequential steps with tool usage.',
        'Map Agent: Applies the same operation across multiple items in parallel.',
        'Squad Agent: Coordinates multiple specialized agents to work together on tasks.',
        'Simple Chat Agent: Provides basic conversational responses without tool usage.',
    ]

    results = await cross_encoder.rank(query=query, texts=agent_descriptions)

    # Get the top-ranked agent
    best_agent = sorted(results, key=lambda x: x[1], reverse=True)[0]
    print(f'Best agent: {best_agent[0][1]} (score: {best_agent[1]:.2f})')
```

This pattern is powerful for:
- **Routing** - Direct users to the right agent automatically
- **Tool selection** - Choose appropriate tools based on task description
- **Dynamic orchestration** - Adapt agent selection based on context

---

## Example 4: Custom Score Range

Customize the scoring scale to match your application needs:

```python
async def custom_score_range():
    # Using 0-100 range instead of default -5 to 5
    cross_encoder = build_cross_encoder(
        'llm',
        llm='openai:gpt-4o-mini',
        score_range=(0, 100)
    )

    query = 'Python programming tutorials'
    texts = [
        'Learn Python from scratch with interactive examples and exercises.',
        'JavaScript is the language of the web for frontend development.',
        'Advanced Python techniques for data science and machine learning.',
    ]

    results = await cross_encoder.rank(query=query, texts=texts)
```

**Output:**
```
Score:  95.0/100 | Learn Python from scratch with interactive examples...
Score:  88.0/100 | Advanced Python techniques for data science...
Score:  25.0/100 | JavaScript is the language of the web...
```

Different score ranges can be useful for:
- **Percentage-based scoring** - Use 0-100 for intuitive percentages
- **Binary classification** - Use 0-1 for probability-like scores
- **Fine-grained ranking** - Use wider ranges like -10 to 10 for more granularity

---

## Building Cross-Encoders

### Method 1: Simple String-Based

```python
cross_encoder = build_cross_encoder('llm', llm='openai:gpt-4o-mini')
```

### Method 2: With Configuration

```python
from tinygent.cross_encoders.llm_cross_encoder import LLMCrossEncoderConfig

config = LLMCrossEncoderConfig(
    llm='openai:gpt-4o-mini',
    score_range=(-5.0, 5.0)
)
cross_encoder = build_cross_encoder(config)
```

### Method 3: With Custom Prompt Template

```python
from tinygent.cross_encoders.llm_cross_encoder import LLMCrossEncoderPromptTemplate
from tinygent.core.prompt import TinyPrompt

custom_prompt = LLMCrossEncoderPromptTemplate(
    ranking=TinyPrompt.UserSystem(
        system='You are a relevance scorer...',
        user='Rate this text...'
    )
)

config = LLMCrossEncoderConfig(
    llm='openai:gpt-4o-mini',
    prompt_template=custom_prompt
)
cross_encoder = build_cross_encoder(config)
```

---

## Use Cases in Agent Systems

Cross-encoders are particularly valuable in the **tinygent** framework for:

### 1. **Multi-Agent Routing**
```python
# User asks a question
user_query = "How do I debug my code?"

# Get all available agents
agents = get_available_agents()  # Returns agent descriptions

# Rank agents by relevance
results = await cross_encoder.rank(user_query, agents)

# Route to the best agent
best_agent = results[0][0][1]
```

### 2. **Tool Selection**
```python
# Agent needs to choose the right tool
task = "Get the current weather in Paris"

# Get tool descriptions
tools = get_tool_descriptions()

# Rank tools by relevance
results = await cross_encoder.rank(task, tools)

# Use the most relevant tool
best_tool = results[0][0][1]
```

### 3. **Memory Retrieval**
```python
# Agent needs to recall relevant context
current_query = "What was discussed about the project timeline?"

# Get past conversation snippets
memory_snippets = get_chat_history()

# Rank by relevance
results = await cross_encoder.rank(current_query, memory_snippets)

# Use top 3 most relevant memories
relevant_context = [r[0][1] for r in sorted(results, key=lambda x: x[1], reverse=True)[:3]]
```

### 4. **Knowledge Graph Traversal**
```python
# Find the most relevant entities or relationships
query = "Companies working on AI agents"

# Get candidate entities from knowledge graph
entities = get_kg_entities()

# Rank by relevance
results = await cross_encoder.rank(query, entities)

# Focus on top results
top_entities = results[:5]
```

---

## Performance Considerations

**LLM-based cross-encoders:**
- **Pros**: More accurate semantic understanding, customizable via prompts
- **Cons**: Slower than traditional cross-encoders, requires API calls, costs money

**Traditional cross-encoders** (not yet implemented in tinygent):
- **Pros**: Fast inference, no API costs, good accuracy
- **Cons**: Fixed model behavior, requires model download

For production systems, consider:
- **Caching** - Cache scores for frequently used query-text pairs
- **Batching** - Process multiple pairs in parallel
- **Hybrid approach** - Use fast embeddings for initial filtering, then cross-encoder for re-ranking

---

## Comparison with Embeddings

| Feature | Embeddings | Cross-Encoders |
|---------|-----------|----------------|
| **Speed** | Fast (cosine similarity) | Slower (LLM evaluation) |
| **Accuracy** | Good for broad matching | Better for nuanced relevance |
| **Query-Text Interaction** | Independent encoding | Joint evaluation |
| **Use Case** | Initial retrieval from large corpus | Re-ranking top candidates |

**Recommended pattern:**
1. Use embeddings to retrieve top 50-100 candidates from large corpus
2. Use cross-encoder to re-rank top candidates for final selection

---

## Advanced: Custom Cross-Encoder Implementation

You can implement your own cross-encoder by inheriting from `AbstractCrossEncoder`:

```python
from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoder

class MyCustomCrossEncoder(AbstractCrossEncoder):
    def __init__(self, model_name: str):
        # Your initialization
        pass

    async def rank(self, query: str, texts: Iterable[str]) -> list[tuple[tuple[str, str], float]]:
        # Your ranking logic
        pass

    async def predict(self, pairs: list[tuple[str, str]]) -> list[tuple[tuple[str, str], float]]:
        # Your prediction logic
        pass
```

---

## Related Examples

* [`examples/embeddings/`](../embeddings/) - Basic embedding generation
* [`examples/agents/react/`](../agents/react/) - Agent that could use cross-encoders for tool selection
* [`examples/memory/`](../memory/) - Memory systems that could benefit from cross-encoder retrieval

---

## Further Reading

* [Cross-Encoders vs Bi-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
* [Using Cross-Encoders for Re-Ranking](https://www.pinecone.io/learn/series/nlp/cross-encoders/)
* [Semantic Search Best Practices](https://www.sbert.net/examples/applications/semantic-search/README.html)
