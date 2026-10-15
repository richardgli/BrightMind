from __future__ import annotations

from langsmith import Client

DATASET_NAME = "brightmind-baseline"
DATASET_DESCRIPTION = (
    "BrightMind agent regression + quality eval. Routing order, retrieval "
    "grounding, and output shape. Source of truth lives in eval/dataset.py."
)

CASES: list[dict] = [
    # --- Routing: quiz requests must research first, then quiz ---
    {
        "question": "Make me a 3-question quiz on binary search trees.",
        "expected_route": ["research_agent", "quiz_agent"],
        "require_retrieval": True,
        "expect_quiz": True,
        "forbid_route": [],
    },
    {
        "question": "Quiz me on the attention mechanism in transformers.",
        "expected_route": ["research_agent", "quiz_agent"],
        "require_retrieval": True,
        "expect_quiz": True,
        "forbid_route": [],
    },
    # --- Routing: explanation requests should research, not quiz ---
    {
        "question": "Explain how a hash table handles collisions.",
        "expected_route": ["research_agent"],
        "require_retrieval": True,
        "expect_quiz": False,
        "forbid_route": ["quiz_agent"],
    },
    {
        "question": "Summarize the key idea of self-attention.",
        "expected_route": ["research_agent"],
        "require_retrieval": True,
        "expect_quiz": False,
        "forbid_route": ["quiz_agent"],
    },
    # --- Output shape: a quiz must contain questions ---
    {
        "question": "Give me 2 practice questions about pointers in C.",
        "expected_route": ["research_agent", "quiz_agent"],
        "require_retrieval": True,
        "expect_quiz": True,
        "forbid_route": [],
    },
]


def build_examples() -> list[dict]:
    """Return examples in the shape create_examples expects."""
    examples = []
    for case in CASES:
        examples.append(
            {
                "inputs": {"question": case["question"]},
                "outputs": {
                    "expected_route": case["expected_route"],
                    "require_retrieval": case["require_retrieval"],
                    "expect_quiz": case["expect_quiz"],
                    "forbid_route": case["forbid_route"],
                },
            }
        )
    return examples


def ensure_dataset(client: Client | None = None) -> str:
    """Create the dataset if missing and (re)seed examples. Skips creation if the dataset already exists. 
    Returns the dataset name."""
    client = client or Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        existing = list(client.list_examples(dataset_name=DATASET_NAME))
        if existing:
            print(
                f"Dataset '{DATASET_NAME}' already exists with "
                f"{len(existing)} examples"
            )
            return DATASET_NAME
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME, description=DATASET_DESCRIPTION
        )
        print(f"Created dataset '{DATASET_NAME}'.")

    client.create_examples(dataset_id=dataset.id, examples=build_examples())
    print(f"Seeded {len(CASES)} examples into '{DATASET_NAME}'.")
    return DATASET_NAME


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    ensure_dataset()
