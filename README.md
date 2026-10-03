# LLM Hallucination & Fact-Checking Benchmarker

A claim-level LLM evaluation framework for detecting hallucinations, verifying factual grounding, identifying contradictions, and benchmarking response quality against reference evidence.

## Overview

Large Language Models can generate responses that appear convincing but contain unsupported, incorrect, or contradictory claims.

This project evaluates LLM-generated responses at the **claim level** rather than treating the entire response as a single unit.

The system:

- Decomposes responses into atomic claims
- Compares claims against supporting evidence
- Detects unsupported and contradictory statements
- Identifies potential hallucinated content
- Calculates evaluation metrics
- Calibrates decision thresholds
- Generates benchmark results and reports
- Includes automated tests for evaluation components

## Key Capabilities

- **Claim-Level Evaluation** — Breaks LLM responses into individual factual claims.
- **Hallucination Detection** — Identifies claims that lack sufficient supporting evidence.
- **Evidence-Based Fact Checking** — Compares generated claims with reference information.
- **Contradiction Detection** — Detects claims that conflict with available evidence.
- **Benchmark Evaluation** — Runs systematic evaluations across datasets.
- **Threshold Calibration** — Helps determine decision thresholds for classification.
- **Response Quality Scoring** — Produces quantitative evaluation signals.
- **Automated Testing** — Includes tests for important evaluation components.
- **Interactive Demo** — Provides a Streamlit interface for evaluating responses.

## Evaluation Workflow

```text
LLM Response
     ↓
Claim Segmentation
     ↓
Atomic Claim Extraction
     ↓
Evidence Retrieval / Comparison
     ↓
Claim Evaluation
     ↓
Supported / Unsupported / Contradictory
     ↓
Hallucination Analysis
     ↓
Metrics & Benchmark Results
     ↓
Evaluation Report
```

## Project Structure

```text
LLM-FactCheck-Benchmarker/
│
├── README.md
├── requirements.txt
├── main.py
├── app.py
├── LLM_Response_Evaluation_Hallucination_Benchmark.ipynb
│
├── src/
│   ├── atomic_claims.py
│   ├── benchmark_runner.py
│   ├── contradiction.py
│   ├── evaluator.py
│   ├── ingestor.py
│   ├── metrics.py
│   ├── reporter.py
│   ├── segmenter.py
│   └── threshold_calibration.py
│
├── tests/
│   ├── test_adversarial.py
│   └── test_atomic_claims.py
│
└── data/
    ├── evaluation_dataset.json
    ├── evaluation_dataset_v2.json
    ├── benchmark_results.csv
    └── threshold_results.csv
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- NLP / LLM Evaluation
- Streamlit
- JSON / CSV
- Pytest
- Google Colab
- Git & GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/rameshwarkawade/LLM_Response_Evaluation_Hallucination_Benchmark.ipynb.git
cd LLM_Response_Evaluation_Hallucination_Benchmark.ipynb
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the benchmark

```bash
python main.py
```

### Run the interactive evaluator

```bash
streamlit run app.py
```

### Notebook

The complete evaluation workflow is also demonstrated in:

```text
LLM_Response_Evaluation_Hallucination_Benchmark.ipynb
```

The notebook provides an interactive environment for running and inspecting the evaluation pipeline.

## Evaluation Outputs

The system produces evaluation artifacts including:

- Benchmark results
- Threshold calibration results
- Evaluation datasets
- Claim-level evaluation signals
- Response quality measurements
- Automated evaluation reports

Results are stored in the `data/` directory where applicable.

## Testing

Run the automated tests with:

```bash
pytest
```

The test suite covers important components including:

- Atomic claim processing
- Adversarial evaluation scenarios
- Evaluation logic

## Why This Project Matters

Reliable evaluation is critical for production LLM systems because fluent responses are not necessarily factually correct.

This project focuses on **measuring and improving LLM response reliability** through:

- Evidence-based evaluation
- Claim-level analysis
- Hallucination detection
- Contradiction analysis
- Benchmarking
- Quantitative quality signals

## Future Improvements

Potential extensions include:

- Multi-model comparison
- Human-vs-LLM evaluation analysis
- Additional hallucination benchmarks
- RAG-specific evaluation
- Semantic similarity analysis
- Advanced evaluator calibration
- Evaluation dashboards
- Continuous evaluation pipelines

## Author

**Rameshwar Kawade**

B.Tech — Civil Engineering  
IIT (ISM) Dhanbad

GitHub: [@rameshwarkawade](https://github.com/rameshwarkawade)
