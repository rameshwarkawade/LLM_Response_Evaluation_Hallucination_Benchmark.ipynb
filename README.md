# LLM Hallucination & Fact-Checking Benchmarker

## Overview

A claim-level evaluation framework for measuring the factual grounding of LLM-generated responses against trusted reference documents.

The system decomposes generated responses into individual claims, compares claims with supporting evidence, identifies unsupported or contradictory statements, and produces quantitative evaluation metrics.

## Problem

Large Language Models can generate responses that sound plausible but contain unsupported, incorrect, or contradictory information.

Evaluating an entire response with a single score can hide these failures.

This project therefore evaluates responses at the claim level.

## Evaluation Pipeline

Reference Document
        ↓
LLM Response
        ↓
Claim Extraction
        ↓
Semantic Evidence Matching
        ↓
Grounding / Contradiction Analysis
        ↓
Claim-Level Verdict
        ↓
Faithfulness & Error Metrics

## Key Capabilities

- Claim-level LLM evaluation
- Semantic evidence matching
- Unsupported claim detection
- Contradiction analysis
- Faithfulness scoring
- Threshold calibration
- Precision, Recall and F1 evaluation
- Confusion matrix analysis
- Error analysis by failure category
- Interactive Streamlit dashboard

## Technology

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- spaCy
- Streamlit

## Evaluation Categories

- Faithful responses
- Unsupported claims
- Contradictions
- Numerical errors
- Entity errors
- Temporal errors
- Negation errors
- Paraphrases
- Mixed responses

## Project Structure

LLM-FactCheck-Benchmarker/

├── data/
├── src/
├── app.py
├── main.py
├── requirements.txt
└── README.md

## Running the Project

Install dependencies:

pip install -r requirements.txt

Run the benchmark:

python main.py

Run the dashboard:

streamlit run app.py

## Evaluation Methodology

The evaluator operates at the claim level rather than assigning a single label to an entire response.

Similarity thresholds are tested across multiple values and evaluated against a reference-labeled benchmark.

Performance is measured using:

- Accuracy
- Precision
- Recall
- F1
- Confusion Matrix

Failure cases are further analyzed by category to identify weaknesses in the evaluation system.

## Limitations

Semantic similarity alone cannot guarantee factual correctness.

A claim can be semantically similar to a source while still containing a changed number, date, entity, or logical relationship.

Therefore, the system should be treated as an evaluation aid rather than a fully autonomous fact-checker.

## Future Improvements

- Larger independently reviewed benchmark
- NLI-based entailment models
- Multi-document evidence retrieval
- Human-in-the-loop review
- Model-to-model benchmarking
- Evaluation of long-form responses
