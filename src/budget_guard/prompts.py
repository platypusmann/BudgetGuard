"""Single source of truth for the prompt. Every tier, including Tier 0 on Kaggle, uses exactly this.

Changing anything here invalidates the cache, which is why PROMPT_VERSION goes into every cache row.
"""

PROMPT_VERSION = "v1"

SYSTEM = (
    "You answer trivia questions. Reply with the short answer only, "
    "such as a name, a date, a number or a short phrase. "
    "No explanation, no full sentence, no punctuation at the end."
)


def user_message(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(question)},
    ]
