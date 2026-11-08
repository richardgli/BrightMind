from __future__ import annotations

from langsmith import Client

from eval.cases.routing_cases import CASES as ROUTING_CASES, DESCRIPTION as ROUTING_DESCRIPTION
from eval.cases.tutor_cases import CASES as TUTOR_CASES, DESCRIPTION as TUTOR_DESCRIPTION

def build_single_turn_examples(key: str) -> list[dict]:
    """Return single turn examples in the shape create_examples expects."""
    spec = DATASETS[key]
    examples = []
    for case in spec["cases"]:
        examples.append(
            {
                "inputs": {"question": case["question"]},
                "outputs": {
                    "expected_route": case["expected_route"],
                    "require_retrieval": case["require_retrieval"],
                    "expect_quiz": case["expect_quiz"],
                    "forbid_route": case["forbid_route"],
                    "expect_socratic": case.get("expect_socratic", False),
                },
            }
        )
    return examples


def build_multiturn_examples(key: str) -> list[dict]:
    """Return multiturn examples in the shape create_examples expects."""
    spec = DATASETS[key]
    examples = []
    for case in spec["cases"]:
        examples.append(
            {
                "inputs": {"turns": case["turns"]},
                "outputs": {
                    "expect_tutor_each_turn": case.get("expect_tutor_each_turn", True),
                    "expect_no_answer_leak": case.get("expect_no_answer_leak", True)
                },
            }
        )
    return examples


DATASETS = {
    "routing": {
        "name": "brightmind-routing-v1",
        "description": ROUTING_DESCRIPTION,
        "cases": ROUTING_CASES,
        "builder": build_single_turn_examples,
    },
    "tutor": {
        "name": "brightmind-tutor-multiturn-v1",
        "description": TUTOR_DESCRIPTION,
        "cases": TUTOR_CASES,
        "builder": build_multiturn_examples,
    },
}

def ensure_dataset(key: str, client: Client | None = None) -> str:
    """Create the dataset if missing and (re)seed examples. Skips creation if the dataset already exists. 
    Returns the dataset name."""
    spec = DATASETS[key]
    client = client or Client()

    if client.has_dataset(dataset_name=spec["name"]):
        existing = list(client.list_examples(dataset_name=spec["name"]))
        if existing:
            print(
                f"Dataset '{spec['name']}' already exists with "
                f"{len(existing)} examples"
            )
            return spec["name"]
        dataset = client.read_dataset(dataset_name=spec["name"])
    else:
        dataset = client.create_dataset(
            dataset_name=spec["name"], description=spec["description"]
        )
        print(f"Created dataset '{spec['name']}'.")

    client.create_examples(dataset_id=dataset.id, examples=spec["builder"](key))
    print(f"Seeded {len(spec['cases'])} examples into '{spec['name']}'.")
    return spec["name"]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    for key in DATASETS:
        ensure_dataset(key)
