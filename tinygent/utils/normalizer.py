def normalize_content(content: str | list[str | dict]) -> str:
    """Normalize content which can be a string or a list of strings and dicts."""
    if isinstance(content, str):
        return content

    return ''.join(
        part if isinstance(part, str) else f'[{part.get("type", "object")}]'
        for part in content
    )
