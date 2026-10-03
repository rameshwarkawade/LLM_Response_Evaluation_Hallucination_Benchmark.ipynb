
from pathlib import Path
import json
import pandas as pd

from src.ingestor import load_dataset
from src.atomic_claims import extract_atomic_claims
from src.evaluator import evaluate_claim
from src.reporter import build_audit_report, calculate_summary


DATASET_PATH = Path(
    "data/evaluation_dataset.json"
)

RESULTS_DIR = Path(
    "data/results"
)


def evaluate_case(case, threshold=0.60):
    """
    Evaluate one source/response pair.
    """

    source = case.get(
        "source",
        ""
    )

    response = case.get(
        "response",
        ""
    )

    claims = extract_atomic_claims(
        response
    )

    results = []

    for claim in claims:

        result = evaluate_claim(
            claim,
            source,
            threshold=threshold
        )

        results.append(
            result
        )

    audit = build_audit_report(
        results
    )

    return audit


def run_pipeline(
    dataset_path=DATASET_PATH,
    threshold=0.60
):
    """
    Run the complete evaluation pipeline.
    """

    print("=" * 65)
    print("LLM FACT-CHECK BENCHMARKER")
    print("=" * 65)

    dataset = load_dataset(
        dataset_path
    )

    print(
        f"Loaded {len(dataset)} evaluation cases."
    )

    all_audits = []

    for index, case in enumerate(
        dataset,
        start=1
    ):

        print(
            f"Evaluating case {index}/{len(dataset)}..."
        )

        audit = evaluate_case(
            case,
            threshold=threshold
        )

        for record in audit:

            record["test_id"] = case.get(
                "test_id",
                f"CASE_{index:03d}"
            )

            all_audits.append(
                record
            )

    audit_df = pd.DataFrame(
        all_audits
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_path = (
        RESULTS_DIR /
        "final_claim_audit.csv"
    )

    audit_df.to_csv(
        audit_path,
        index=False
    )

    if not audit_df.empty:

        summary = calculate_summary(
            all_audits
        )

    else:

        summary = {
            "total_claims": 0,
            "grounded_claims": 0,
            "contradictions": 0,
            "unsupported_claims": 0,
            "needs_review": 0,
            "faithfulness_score": 0.0
        }

    report_path = (
        RESULTS_DIR /
        "final_benchmark_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)

    print(
        f"Claims evaluated: {summary['total_claims']}"
    )

    print(
        f"Faithfulness: {summary['faithfulness_score']}%"
    )

    print(
        f"Contradictions: {summary['contradictions']}"
    )

    print(
        f"Unsupported: {summary['unsupported_claims']}"
    )

    print(
        f"Needs review: {summary['needs_review']}"
    )

    print(
        f"Audit saved to: {audit_path}"
    )

    return audit_df, summary


if __name__ == "__main__":

    run_pipeline()
