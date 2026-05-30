from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful retrieval-augmented assistant. "
    "Answer the question using only the provided context. "
    "If the context does not contain enough information, say that the answer is not available in the provided context."
)

DEFAULT_USER_TEMPLATE = """Question:
{query}

Context:
{context}

Answer:"""


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def build_rag_prompt(
    query: str,
    context: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
    include_system_prompt: bool = True,
) -> str:
    _require_non_empty_str(query, "query")
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    _require_non_empty_str(system_prompt, "system_prompt")
    _require_non_empty_str(user_template, "user_template")
    if not isinstance(include_system_prompt, bool):
        raise ValueError("include_system_prompt must be a boolean")
    if "{query}" not in user_template:
        raise ValueError("user_template must include {query}")
    if "{context}" not in user_template:
        raise ValueError("user_template must include {context}")

    filled = user_template.format(query=query, context=context)
    if include_system_prompt:
        return f"System:\n{system_prompt}\n\nUser:\n{filled}"
    return filled


def build_chat_messages(
    query: str,
    context: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    _require_non_empty_str(query, "query")
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    _require_non_empty_str(system_prompt, "system_prompt")
    user_content = DEFAULT_USER_TEMPLATE.format(query=query, context=context)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
