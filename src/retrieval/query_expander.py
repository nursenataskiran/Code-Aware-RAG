"""
Query expansion for bridging the natural-language ↔ code vocabulary gap.

Expands user queries by:
1. Extracting likely code identifiers (class names, function names)
2. Mapping common terms to project-specific vocabulary
3. Generating multiple search queries for better recall
"""

from __future__ import annotations

import re
from typing import List


# ── Project name aliases ─────────────────────────────────────────────
# Maps common query terms to project_name values in the vector store.
# Extend this as you add more projects.
PROJECT_ALIASES = {
    "f1": "Formula1_Race_Prediction",
    "formula 1": "Formula1_Race_Prediction",
    "formula1": "Formula1_Race_Prediction",
    "race prediction": "Formula1_Race_Prediction",
    "race": "Formula1_Race_Prediction",
    "stress": "Human_Stress_Prediction",
    "human stress": "Human_Stress_Prediction",
    "stress prediction": "Human_Stress_Prediction",
}

# ── Code vocabulary ──────────────────────────────────────────────────
# Maps natural language concepts to likely code identifiers.
CODE_VOCABULARY = {
    "data cleaning": ["data_cleaning", "clean_data", "SessionData", "F1DataCleaner"],
    "data preparation": ["LSTMDataPrep", "LSTMDataPreparer", "DataPreprocessor", "prepare"],
    "feature extraction": ["extract_features", "FeatureExtractor", "num_features"],
    "model": ["LSTM", "StressModel", "model", "Model"],
    "training": ["model.fit", "train", "epochs", "batch_size"],
    "prediction": ["predict", "StressPredictor", "y_pred"],
    "preprocessing": ["DataPreprocessor", "TextCleaner", "clean", "preprocess"],
    "text cleaning": ["TextCleaner", "clean", "clean_text"],
    "loading data": ["DataLoader", "load_data", "load_from_pkl"],
    "lstm": ["LSTM", "LSTMDataPreparer", "TimeDistributed"],
    "normalization": ["normalize_columns", "MinMaxScaler", "scaler"],
    "missing data": ["complete_missing_laps", "fillna", "bfill", "ffill"],
    "tyre": ["TyreLife", "tyre_life", "Compound", "FreshTyre"],
    "position": ["Position", "fill_position", "final_position"],
    "lap": ["LapTime", "LapNumber", "max_laps", "driver_laps"],
    "hungarian algorithm": ["linear_sum_assignment", "cost_matrix", "Hungarian"],
}


def expand_query(query: str) -> List[str]:
    """
    Expand a user query into multiple search queries.

    Returns:
        A list of queries. The first is always the original query.
        Additional queries inject code-specific keywords.
    """
    queries = [query]

    query_lower = query.lower()

    # ── Inject code vocabulary ────────────────────────────────────────
    injected_terms = []
    for concept, code_terms in CODE_VOCABULARY.items():
        if concept in query_lower:
            injected_terms.extend(code_terms)

    # ── Extract potential identifiers already in the query ────────────
    # Match CamelCase or snake_case patterns the user may have typed
    existing_identifiers = re.findall(
        r'[A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:_[a-z]+)+',
        query,
    )

    if injected_terms:
        # Build an expanded query with code keywords
        unique_terms = list(dict.fromkeys(injected_terms))  # preserve order, dedupe
        expanded = f"{query} {' '.join(unique_terms[:6])}"
        queries.append(expanded)

    return queries


def detect_project(query: str) -> str | None:
    """
    Detect which project the query is about, based on keyword matching.
    Returns the project_name string suitable for metadata filtering.
    """
    query_lower = query.lower()
    for alias, project_name in PROJECT_ALIASES.items():
        if alias in query_lower:
            return project_name
    return None
