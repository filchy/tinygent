from tinygent.core.factory import build_embedder


def _print_res(embeddings: list[float], n: int = 3) -> str:
    head = ', '.join(map(str, embeddings[:n]))
    tail = ', '.join(map(str, embeddings[-n:]))
    return f'[{head}, ..., {tail}]'


def single_embed():
    embedder = build_embedder('gemini:gemini-embedding-001')

    embs = embedder.embed(
        'TinyGent is the greatest and tyniest framework in the whole world!'
    )
    print(f'[SINGLE EMBEDDING] len: {len(embs)} | embeddings: {_print_res(embs)}')


def batch_embed():
    embedder = build_embedder('gemini:gemini-embedding-001')

    embs = embedder.embed_batch(
        [
            'TinyGent is the greatest and tyniest framework in the whole world!',
            'LangChain sucks baby.',
            'Yep, i rly said that ;)',
        ]
    )

    print(f'[BATCH EMBEDDINGS]: embs: {len(embs)}')
    print('\n'.join(f'\tsingle: {len(e)} {_print_res(e)}' for e in embs))


def main():
    single_embed()
    batch_embed()


if __name__ == '__main__':
    main()
