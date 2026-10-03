
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.atomic_claims import extract_atomic_claims
from src.evaluator import evaluate_claim
from src.reporter import build_audit_report


st.set_page_config(
    page_title="LLM Fact-Check Benchmarker",
    page_icon="✓",
    layout="wide"
)


RESULTS_DIR = Path(
    "data/results"
)


st.title(
    "LLM Fact-Check Benchmarker"
)

st.caption(
    "Claim-level hallucination and grounding evaluation"
)


def load_results():

    results_file = (
        RESULTS_DIR /
        "final_claim_audit.csv"
    )

    if not results_file.exists():
        return pd.DataFrame()

    return pd.read_csv(
        results_file
    )


def load_metrics():

    metrics_file = (
        RESULTS_DIR /
        "final_benchmark_report.json"
    )

    if not metrics_file.exists():
        return {}

    with open(
        metrics_file,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_threshold_results():

    file_path = (
        RESULTS_DIR /
        "threshold_summary.csv"
    )

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        file_path
    )


benchmark_df = load_results()
report = load_metrics()
threshold_df = load_threshold_results()


metrics = report.get(
    "metrics",
    {}
)


st.sidebar.header(
    "Navigation"
)

page = st.sidebar.radio(
    "Select view",
    [
        "Overview",
        "Claim Audit",
        "Error Analysis",
        "Threshold Analysis",
        "Live Evaluation"
    ]
)


if page == "Overview":

    st.header(
        "Evaluation Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Claims",
        metrics.get(
            "total_claims",
            0
        )
    )

    col2.metric(
        "Faithfulness",
        f'{metrics.get("faithfulness_score", 0)}%'
    )

    col3.metric(
        "Contradictions",
        metrics.get(
            "contradictions",
            0
        )
    )

    col4.metric(
        "Human Review",
        metrics.get(
            "needs_review",
            0
        )
    )

    st.divider()

    if not benchmark_df.empty:

        st.subheader(
            "Verdict Distribution"
        )

        verdict_counts = (
            benchmark_df[
                "verdict"
            ]
            .value_counts()
        )

        st.bar_chart(
            verdict_counts
        )

        st.subheader(
            "Recent Claim Audits"
        )

        st.dataframe(
            benchmark_df[
                [
                    "claim",
                    "evidence",
                    "similarity_score",
                    "verdict",
                    "error_type"
                ]
            ].head(10),
            use_container_width=True
        )

    else:

        st.warning(
            "No benchmark results found."
        )


elif page == "Claim Audit":

    st.header(
        "Claim-Level Evidence Audit"
    )

    if benchmark_df.empty:

        st.warning(
            "Run the benchmark before opening the audit."
        )

    else:

        verdict_filter = st.multiselect(
            "Filter by verdict",
            options=sorted(
                benchmark_df[
                    "verdict"
                ].dropna().unique()
            )
        )

        error_filter = st.multiselect(
            "Filter by error type",
            options=sorted(
                benchmark_df[
                    "error_type"
                ].dropna().unique()
            )
        )

        filtered_df = benchmark_df.copy()

        if verdict_filter:

            filtered_df = filtered_df[
                filtered_df["verdict"].isin(
                    verdict_filter
                )
            ]

        if error_filter:

            filtered_df = filtered_df[
                filtered_df["error_type"].isin(
                    error_filter
                )
            ]

        st.write(
            f"{len(filtered_df)} claim records"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )


elif page == "Error Analysis":

    st.header(
        "Error Taxonomy Analysis"
    )

    if benchmark_df.empty:

        st.warning(
            "No benchmark data available."
        )

    else:

        error_counts = (
            benchmark_df[
                "error_type"
            ]
            .value_counts()
        )

        st.bar_chart(
            error_counts
        )

        st.subheader(
            "Severity Distribution"
        )

        severity_counts = (
            benchmark_df[
                "severity"
            ]
            .value_counts()
        )

        st.bar_chart(
            severity_counts
        )

        st.subheader(
            "Error Records"
        )

        st.dataframe(
            benchmark_df[
                [
                    "claim",
                    "evidence",
                    "error_type",
                    "severity",
                    "reason"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


elif page == "Threshold Analysis":

    st.header(
        "Similarity Threshold Calibration"
    )

    if threshold_df.empty:

        st.warning(
            "Threshold results are not available."
        )

    else:

        chart_df = threshold_df.set_index(
            "threshold"
        )[
            [
                "precision",
                "recall",
                "f1"
            ]
        ]

        st.line_chart(
            chart_df
        )

        st.subheader(
            "Threshold Results"
        )

        st.dataframe(
            threshold_df,
            use_container_width=True,
            hide_index=True
        )


elif page == "Live Evaluation":

    st.header(
        "Live Claim Evaluation"
    )

    st.write(
        "Enter a reference source and an LLM response "
        "to evaluate individual claims."
    )

    source_text = st.text_area(
        "Reference Source",
        height=220,
        placeholder=(
            "Paste the trusted source document here..."
        )
    )

    response_text = st.text_area(
        "LLM Response",
        height=220,
        placeholder=(
            "Paste the generated response here..."
        )
    )

    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.40,
        max_value=0.90,
        value=0.60,
        step=0.05
    )

    if st.button(
        "Evaluate Response"
    ):

        if not source_text.strip():

            st.error(
                "Please provide a reference source."
            )

        elif not response_text.strip():

            st.error(
                "Please provide an LLM response."
            )

        else:

            claims = extract_atomic_claims(
                response_text
            )

            results = []

            for claim in claims:

                result = evaluate_claim(
                    claim,
                    source_text,
                    threshold=threshold
                )

                results.append(
                    result
                )

            audit_report = build_audit_report(
                results
            )

            if audit_report:

                live_df = pd.DataFrame(
                    audit_report
                )

                st.subheader(
                    "Claim-Level Results"
                )

                st.dataframe(
                    live_df,
                    use_container_width=True,
                    hide_index=True
                )

                grounded = sum(
                    item["verdict"] == "grounded"
                    for item in audit_report
                )

                total = len(
                    audit_report
                )

                faithfulness = (
                    grounded / total * 100
                    if total
                    else 0
                )

                st.metric(
                    "Response Faithfulness",
                    f"{faithfulness:.2f}%"
                )

            else:

                st.warning(
                    "No claims were extracted from the response."
                )
