# Embeddings Example

This example demonstrates how to use a custom Embedder implementation that conforms to the `AbstractEmbedder` interface in **tinygent**. It shows basic usage of different embedding methods, including:

* Single text embedding
* Batch text embeddings

## Quick Start

```bash
uv sync --extra openai

uv run examples/embeddings/main.py
```

---

## Requirements

* You must implement an Embedder class inheriting from `AbstractEmbedder`
* Your Embedder must support both synchronous and asynchronous embedding methods
* The embedder is created using the `build_embedder()` factory function

---

## Interface: `AbstractEmbedder`

The `AbstractEmbedder` defines the following required methods:

```python
class AbstractEmbedder(ABC):

    def embed(query: str) -> list[float]
    async def aembed(query: str) -> list[float]

    def embed_batch(queries: list[str]) -> list[list[float]]
    async def aembed_batch(queries: list[str]) -> list[list[float]]
```

Each method returns embedding vectors as lists of floats. The dimensionality depends on the model used.

---

## Example: Single Text Embedding

```python
def single_embed():
    embedder = build_embedder('openai:text-embedding-3-small')
    
    embs = embedder.embed('TinyGent is the greatest and tyniest framework in the whole world!')
    print(f'[SINGLE EMBEDDING] len: {len(embs)} | embeddings: {embs[:3]}...')
```

---

## Example: Batch Text Embeddings

```python
def batch_embed():
    embedder = build_embedder('openai:text-embedding-3-small')
    
    embs = embedder.embed_batch([
        'TinyGent is the greatest and tyniest framework in the whole world!',
        'LangChain sucks baby.',
        'Yep, i rly said that ;)',
    ])
    
    print(f'[BATCH EMBEDDINGS]: embs: {len(embs)}')
    for e in embs:
        print(f'\tsingle: {len(e)} embeddings')
```

---

## Running the Example

```bash
uv run examples/embeddings/main.py
```

Expected output:

```
[SINGLE EMBEDDING] len: 1536 | embeddings: [0.123, -0.456, 0.789, ..., 0.321, -0.654, 0.987]
[BATCH EMBEDDINGS]: embs: 3
	single: 1536 [0.123, -0.456, 0.789, ..., 0.321, -0.654, 0.987]
	single: 1536 [-0.234, 0.567, -0.890, ..., 0.432, 0.765, -0.098]
	single: 1536 [0.345, 0.678, 0.901, ..., -0.543, 0.876, 0.109]
```

---

---

## Notes

* The `build_embedder()` factory function uses the format `provider:model-name` (e.g., `openai:text-embedding-3-small`)
* Embeddings are returned as `list[float]` with dimensionality depending on the model (e.g., 1536 for `text-embedding-3-small`)
* Both synchronous (`embed`, `embed_batch`) and asynchronous (`aembed`, `aembed_batch`) methods are available
* Batch processing is more efficient for multiple texts than calling `embed()` repeatedly
* Common use cases: semantic search, clustering, recommendations, RAG systems, and classification
* The Embedder implementation handles provider-specific API details internally
