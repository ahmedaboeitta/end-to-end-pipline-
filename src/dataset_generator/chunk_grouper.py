import json


def load_chunks(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        return json.load(f)


def group_chunks(chunks: list[dict], max_tokens: int = 500) -> list[list[dict]]:
    """Group chunks where total token count <= max_tokens."""
    groups = []
    current_group = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = chunk["token_count"]

        if current_tokens + chunk_tokens <= max_tokens:
            current_group.append(chunk)
            current_tokens += chunk_tokens
        else:
            if current_group:
                groups.append(current_group)
            current_group = [chunk]
            current_tokens = chunk_tokens

    if current_group:
        groups.append(current_group)

    return groups


def format_sections(group: list[dict]) -> str:
    """Format group of chunks into sections string for prompt."""
    sections = []
    for i, chunk in enumerate(group, 1):
        sections.append(f"Section {i}: {chunk['section_title']}\n{chunk['content']}")
    return "\n\n".join(sections)