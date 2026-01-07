from hashlib import blake2b
import math
import os
import re

_TINY_MINHASH_PERMUTATIONS = int(os.getenv('TINY_GRAPH_MINHASH_PERMUTATIONS', 32))
_TINY_MINHASH_BAND_SIZE = int(os.getenv('TINY_GRAPH_MINHASH_BAND_SIZE', 4))
_TINY_MIN_NAME_LENGTH = int(os.getenv('TINY_GRAPH_MIN_NAME_LENGTH', 6))
_TINY_MIN_TOKEN_COUNT = int(os.getenv('TINY_GRAPH_MIN_TOKEN_COUNT', 2))
_TINY_NAME_ENTROPY_THRESHOLD = float(os.getenv('TINY_GRAPH_NAME_ENTROPY_THRESHOLD', 1.5))
_TINY_FUZZY_JACCARD_THRESHOLD = float(
    os.getenv('TINY_GRAPH_FUZZY_JACCARD_THRESHOLD', 0.8)
)


__all__ = [
    '_TINY_FUZZY_JACCARD_THRESHOLD',
    'has_high_entropy',
    'normalize_string_exact',
    'normalize_string_for_fuzzy',
    'shingles',
    'minhash_signature',
    'lsh_bands',
    'jaccard_similarity',
]


def _hash(text: str, seed: int) -> int:
    """Generate a deterministic 64-bit hash for a shingle given the permutation seed."""
    digest = blake2b(f'{seed}:{text}'.encode(), digest_size=8)
    return int.from_bytes(digest.digest(), 'big')


def _name_entropy(normalized_name: str) -> float:
    """Approximate text specificity using Shannon entropy over characters.

    We strip spaces, count how often each character appears, and sum
    probability * -log2(probability). Short or repetitive names yield low
    entropy, which signals we should defer resolution to the LLM instead of
    trusting fuzzy similarity.
    """
    if not normalized_name:
        return 0.0

    counts: dict[str, int] = {}
    for char in normalized_name.replace(' ', ''):
        counts[char] = counts.get(char, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def has_high_entropy(normalized_name: str) -> bool:
    """Filter out very short or low-entropy names that are unreliable for fuzzy matching."""
    token_count = len(normalized_name.split())
    if (
        len(normalized_name) < _TINY_MIN_NAME_LENGTH
        and token_count < _TINY_MIN_TOKEN_COUNT
    ):
        return False

    return _name_entropy(normalized_name) >= _TINY_NAME_ENTROPY_THRESHOLD


def normalize_string_exact(name: str) -> str:
    """Lowercase text and collapse whitespace so equal names map to the same key."""
    normalized = re.sub(r'[\s]+', ' ', name.lower())
    return normalized.strip()


def normalize_string_for_fuzzy(name: str) -> str:
    """Produce a fuzzier form that keeps alphanumerics and apostrophes for n-gram shingles."""
    normalized = re.sub(r"[^a-z0-9' ]", ' ', normalize_string_exact(name))
    normalized = normalized.strip()
    return re.sub(r'[\s]+', ' ', normalized)


def shingles(normalized_name: str, n: int = 3) -> set[str]:
    """Create n-gram shingles from the normalized name for MinHash calculations."""
    cleaned = normalized_name.replace(' ', '')
    if not cleaned:
        return set()

    if len(cleaned) < n:
        return {cleaned} if cleaned else set()

    return {cleaned[i : i + n] for i in range(len(cleaned) - 2)}


def minhash_signature(shingles: set[str]) -> list[int]:
    """Compute the MinHash signature for the shingle set across predefined permutations."""
    signature = []

    for i in range(_TINY_MINHASH_PERMUTATIONS):
        min_hash = min(_hash(s, i) for s in shingles)
        signature.append(min_hash)

    return signature


def lsh_bands(signature: list[int]) -> list[tuple[int, ...]]:
    """Split the MinHash signature into fixed-size bands for locality-sensitive hashing."""
    signature_list = list(signature)
    if not signature_list:
        return []

    bands: list[tuple[int, ...]] = []
    for start in range(0, len(signature_list), _TINY_MINHASH_BAND_SIZE):
        band = tuple(signature_list[start : start + _TINY_MINHASH_BAND_SIZE])
        if len(band) == _TINY_MINHASH_BAND_SIZE:
            bands.append(band)
    return bands


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Return the Jaccard similarity between two shingle sets, handling empty edge cases."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union else 0.0


if __name__ == '__main__':
    a = 'tinygent is the best agentic library ever'
    b = 'tinygent is the best agent library ever'

    a_norm = normalize_string_for_fuzzy(a)
    b_norm = normalize_string_for_fuzzy(b)

    print(f'a normilized text: {a_norm}')
    print(f'b normilized text: {b_norm}')

    a_shingles = shingles(a_norm)
    b_shingles = shingles(b_norm)

    print(f'a shingles: {a_shingles}')
    print(f'b shingles: {b_shingles}')

    jac = jaccard_similarity(a_shingles, b_shingles)
    print(f'jaccard similarity: {jac}')

    a_sig = minhash_signature(a_shingles)
    b_sig = minhash_signature(b_shingles)

    print(f'a minhash sig: {a_sig}')
    print(f'b minhash sig: {b_sig}')

    for band_idx, band in enumerate(lsh_bands(a_sig)):
        print(f'a singlebucket: {(band_idx, band)}')
