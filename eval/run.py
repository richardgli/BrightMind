from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

from eval.dataset import ensure_dataset
from eval.target import make_target, make_multiturn_target
from eval.evaluators import (
    DETERMINISTIC_EVALUATORS,
    ALL_EVALUATORS,
    MULTITURN_EVALUATORS,
)


def get_baseline_agent():
    from agent.agent import build_stategraph
    return build_stategraph()


def main() -> None:
    parser = argparse.ArgumentParser(description="BrightMind LangSmith eval harness")
    parser.add_argument("--seed-only", action="store_true",
                        help="only create/seed the dataset, don't run eval")
    parser.add_argument("--no-judge", action="store_true",
                        help="only run deterministic evaluators)")
    parser.add_argument("--prefix", default="supervisor-baseline",
                        help="set LangSmith experiment name prefix")
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--suite", choices=["routing", "tutor"], default="routing")
    args = parser.parse_args()

    load_dotenv()
    client = Client()

    dataset_name = ensure_dataset(args.suite, client)
    if args.seed_only:
        return

    agent = get_baseline_agent()
    if args.suite == "routing":
        target = make_target(agent, structured=True)
        evaluators = DETERMINISTIC_EVALUATORS if args.no_judge else ALL_EVALUATORS
    else:
        target = make_multiturn_target(agent)
        evaluators = MULTITURN_EVALUATORS

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=args.prefix,
        max_concurrency=args.max_concurrency,
        client=client,
        metadata={"system": "stategraph", "judge": not args.no_judge},
    )

    print("\n=== Evaluation summary ===")
    totals: dict[str, list[float]] = {}
    try:
        for row in results:
            eval_results = row["evaluation_results"]["results"]
            for res in eval_results:
                key = res.key
                score = res.score
                if score is None:
                    continue
                totals.setdefault(key, []).append(float(score))
    except Exception as e:
        print(f"(summary unavailable: {e})")
        return

    if not totals:
        print("(no scores captured)")
    for key in sorted(totals):
        scores = totals[key]
        mean = sum(scores) / len(scores)
        print(f"{key:22} mean={mean:.2f}  n={len(scores)}")


if __name__ == "__main__":
    main()
