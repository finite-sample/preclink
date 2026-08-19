#!/usr/bin/env python3
"""Benchmark different record linkage strategies.

This script evaluates preclink's performance across:
- Decision methods (hungarian, greedy, row_sequential)
- String comparison algorithms (jaro_winkler, levenshtein, damerau_levenshtein)
- TF-IDF vs standard string comparison
- Blocking vs full cartesian product
- Various threshold values

Run with: python examples/benchmark_strategies.py
"""

import random
import string
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from preclink import (
    ExactComparison,
    Pipeline,
    StringComparison,
)
from preclink.score.comparisons import TFIDFStringComparison


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    strategy: str
    precision: float
    recall: float
    f1: float
    runtime_ms: float
    n_matches: int


def generate_benchmark_data(
    n_records: int = 500,
    n_duplicates: int = 100,
    noise_level: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, set[tuple[str, str]]]:
    """Generate synthetic benchmark data with known ground truth.

    Args:
        n_records: Number of records in the source dataset.
        n_duplicates: Number of duplicate records to create.
        noise_level: Probability of applying noise to each field.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (source_df, target_df, ground_truth_pairs).
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    first_names = [
        "John",
        "Jane",
        "Robert",
        "Mary",
        "Michael",
        "Patricia",
        "William",
        "Jennifer",
        "David",
        "Linda",
        "James",
        "Elizabeth",
        "Richard",
        "Barbara",
        "Joseph",
        "Susan",
        "Thomas",
        "Jessica",
        "Charles",
        "Sarah",
        "Christopher",
        "Karen",
        "Daniel",
        "Nancy",
        "Matthew",
        "Betty",
        "Anthony",
        "Margaret",
        "Mark",
        "Sandra",
        "Donald",
        "Ashley",
        "Steven",
        "Kimberly",
        "Paul",
        "Emily",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
        "Hernandez",
        "Lopez",
        "Gonzalez",
        "Wilson",
        "Anderson",
        "Thomas",
        "Taylor",
        "Moore",
        "Jackson",
        "Martin",
        "Lee",
        "Perez",
        "Thompson",
        "White",
        "Harris",
        "Sanchez",
        "Clark",
        "Ramirez",
    ]
    cities = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
    ]
    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]

    def apply_noise(value: str, rng: random.Random) -> str:
        """Apply random noise to a string value."""
        if not value or rng.random() > noise_level:
            return value

        noise_type = rng.choice(["typo", "swap", "delete", "case"])
        chars = list(value)

        if noise_type == "typo" and len(chars) > 0:
            pos = rng.randint(0, len(chars) - 1)
            chars[pos] = rng.choice(string.ascii_lowercase)
        elif noise_type == "swap" and len(chars) > 1:
            pos = rng.randint(0, len(chars) - 2)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif noise_type == "delete" and len(chars) > 2:
            pos = rng.randint(0, len(chars) - 1)
            chars.pop(pos)
        elif noise_type == "case":
            return value.upper() if rng.random() > 0.5 else value.lower()

        return "".join(chars)

    records = [
        {
            "id": f"src-{i:04d}",
            "first_name": rng.choice(first_names),
            "last_name": rng.choice(last_names),
            "dob": (
                f"19{rng.randint(50, 99)}-"
                f"{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            ),
            "city": rng.choice(cities),
            "state": rng.choice(states),
        }
        for i in range(n_records)
    ]

    source_df = pd.DataFrame(records)

    duplicates = []
    ground_truth = set()
    dup_indices = np_rng.choice(
        n_records, size=min(n_duplicates, n_records), replace=False
    )

    for idx in dup_indices:
        orig = records[idx].copy()
        dup = {
            "id": f"tgt-{idx:04d}",
            "first_name": apply_noise(orig["first_name"], rng),
            "last_name": apply_noise(orig["last_name"], rng),
            "dob": orig["dob"] if rng.random() > 0.1 else None,
            "city": apply_noise(orig["city"], rng),
            "state": orig["state"],
        }
        duplicates.append(dup)
        ground_truth.add((orig["id"], dup["id"]))

    non_dup_count = n_duplicates // 2
    duplicates.extend(
        {
            "id": f"tgt-new-{i:04d}",
            "first_name": rng.choice(first_names),
            "last_name": rng.choice(last_names),
            "dob": (
                f"19{rng.randint(50, 99)}-"
                f"{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            ),
            "city": rng.choice(cities),
            "state": rng.choice(states),
        }
        for i in range(non_dup_count)
    )

    target_df = pd.DataFrame(duplicates)

    return source_df, target_df, ground_truth


def compute_metrics(
    found_pairs: set[tuple[str, str]],
    ground_truth: set[tuple[str, str]],
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score."""
    if not found_pairs:
        return 0.0, 0.0, 0.0

    true_positives = len(found_pairs & ground_truth)
    precision = true_positives / len(found_pairs) if found_pairs else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def benchmark_decision_methods(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth: set[tuple[str, str]],
) -> list[BenchmarkResult]:
    """Benchmark different decision methods."""
    results = []

    for method in ["hungarian", "greedy", "row_sequential"]:
        pipeline = (
            Pipeline()
            .preprocess(lowercase=True)
            .score(
                [
                    StringComparison(
                        "first_name", algorithm="jaro_winkler", weight=2.0
                    ),
                    StringComparison("last_name", algorithm="jaro_winkler", weight=2.0),
                    ExactComparison("dob", weight=1.0),
                ]
            )
            .filter(min_score=0.7)
            .decide(method=method)
            .build()
        )

        start = time.perf_counter()
        result = pipeline.link(source_df, target_df)
        runtime_ms = (time.perf_counter() - start) * 1000

        found_pairs = set(
            zip(result.matches["id_left"], result.matches["id_right"], strict=True)
        )

        precision, recall, f1 = compute_metrics(found_pairs, ground_truth)

        results.append(
            BenchmarkResult(
                strategy=f"decide:{method}",
                precision=precision,
                recall=recall,
                f1=f1,
                runtime_ms=runtime_ms,
                n_matches=len(found_pairs),
            )
        )

    return results


def benchmark_string_algorithms(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth: set[tuple[str, str]],
) -> list[BenchmarkResult]:
    """Benchmark different string comparison algorithms."""
    results = []

    for algorithm in ["jaro_winkler", "levenshtein", "damerau_levenshtein"]:
        pipeline = (
            Pipeline()
            .preprocess(lowercase=True)
            .score(
                [
                    StringComparison("first_name", algorithm=algorithm, weight=2.0),
                    StringComparison("last_name", algorithm=algorithm, weight=2.0),
                    ExactComparison("dob", weight=1.0),
                ]
            )
            .filter(min_score=0.7)
            .build()
        )

        start = time.perf_counter()
        result = pipeline.link(source_df, target_df)
        runtime_ms = (time.perf_counter() - start) * 1000

        found_pairs = set(
            zip(result.matches["id_left"], result.matches["id_right"], strict=True)
        )

        precision, recall, f1 = compute_metrics(found_pairs, ground_truth)

        results.append(
            BenchmarkResult(
                strategy=f"algo:{algorithm}",
                precision=precision,
                recall=recall,
                f1=f1,
                runtime_ms=runtime_ms,
                n_matches=len(found_pairs),
            )
        )

    return results


def benchmark_tfidf_vs_standard(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth: set[tuple[str, str]],
) -> list[BenchmarkResult]:
    """Benchmark TF-IDF weighted vs standard string comparison."""
    results = []

    pipeline_standard = (
        Pipeline()
        .preprocess(lowercase=True)
        .score(
            [
                StringComparison("first_name", algorithm="jaro_winkler", weight=2.0),
                StringComparison("last_name", algorithm="jaro_winkler", weight=2.0),
            ]
        )
        .filter(min_score=0.7)
        .build()
    )

    start = time.perf_counter()
    result = pipeline_standard.link(source_df, target_df)
    runtime_ms = (time.perf_counter() - start) * 1000

    found_pairs = set(
        zip(result.matches["id_left"], result.matches["id_right"], strict=True)
    )

    precision, recall, f1 = compute_metrics(found_pairs, ground_truth)

    results.append(
        BenchmarkResult(
            strategy="standard",
            precision=precision,
            recall=recall,
            f1=f1,
            runtime_ms=runtime_ms,
            n_matches=len(found_pairs),
        )
    )

    tfidf_first = TFIDFStringComparison(
        "first_name", algorithm="jaro_winkler", weight=2.0
    )
    tfidf_last = TFIDFStringComparison(
        "last_name", algorithm="jaro_winkler", weight=2.0
    )
    tfidf_first.set_idf_weights(source_df["first_name"], target_df["first_name"])
    tfidf_last.set_idf_weights(source_df["last_name"], target_df["last_name"])

    pipeline_tfidf = (
        Pipeline()
        .preprocess(lowercase=True)
        .score([tfidf_first, tfidf_last])
        .filter(min_score=0.5)
        .build()
    )

    start = time.perf_counter()
    result = pipeline_tfidf.link(source_df, target_df)
    runtime_ms = (time.perf_counter() - start) * 1000

    found_pairs = set(
        zip(result.matches["id_left"], result.matches["id_right"], strict=True)
    )

    precision, recall, f1 = compute_metrics(found_pairs, ground_truth)

    results.append(
        BenchmarkResult(
            strategy="tfidf",
            precision=precision,
            recall=recall,
            f1=f1,
            runtime_ms=runtime_ms,
            n_matches=len(found_pairs),
        )
    )

    return results


def benchmark_blocking(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth: set[tuple[str, str]],
) -> list[BenchmarkResult]:
    """Benchmark with and without blocking."""
    results = []

    pipeline_no_block = (
        Pipeline()
        .preprocess(lowercase=True)
        .score(
            [
                StringComparison("first_name", algorithm="jaro_winkler", weight=2.0),
                StringComparison("last_name", algorithm="jaro_winkler", weight=2.0),
            ]
        )
        .filter(min_score=0.7)
        .build()
    )

    start = time.perf_counter()
    result = pipeline_no_block.link(source_df, target_df)
    runtime_ms = (time.perf_counter() - start) * 1000

    found_pairs = set(
        zip(result.matches["id_left"], result.matches["id_right"], strict=True)
    )

    precision, recall, f1 = compute_metrics(found_pairs, ground_truth)

    results.append(
        BenchmarkResult(
            strategy="no_blocking",
            precision=precision,
            recall=recall,
            f1=f1,
            runtime_ms=runtime_ms,
            n_matches=len(found_pairs),
        )
    )

    pipeline_blocked = (
        Pipeline()
        .preprocess(lowercase=True)
        .block(on="state")
        .score(
            [
                StringComparison("first_name", algorithm="jaro_winkler", weight=2.0),
                StringComparison("last_name", algorithm="jaro_winkler", weight=2.0),
            ]
        )
        .filter(min_score=0.7)
        .build()
    )

    start = time.perf_counter()
    result = pipeline_blocked.link(source_df, target_df)
    runtime_ms = (time.perf_counter() - start) * 1000

    found_pairs = set(
        zip(result.matches["id_left"], result.matches["id_right"], strict=True)
    )

    precision, recall, f1 = compute_metrics(found_pairs, ground_truth)

    results.append(
        BenchmarkResult(
            strategy="blocked:state",
            precision=precision,
            recall=recall,
            f1=f1,
            runtime_ms=runtime_ms,
            n_matches=len(found_pairs),
        )
    )

    return results


def benchmark_thresholds(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth: set[tuple[str, str]],
) -> list[BenchmarkResult]:
    """Benchmark different threshold values."""
    results = []

    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pipeline = (
            Pipeline()
            .preprocess(lowercase=True)
            .score(
                [
                    StringComparison(
                        "first_name", algorithm="jaro_winkler", weight=2.0
                    ),
                    StringComparison("last_name", algorithm="jaro_winkler", weight=2.0),
                    ExactComparison("dob", weight=1.0),
                ]
            )
            .filter(min_score=threshold)
            .build()
        )

        start = time.perf_counter()
        result = pipeline.link(source_df, target_df)
        runtime_ms = (time.perf_counter() - start) * 1000

        found_pairs = set(
            zip(result.matches["id_left"], result.matches["id_right"], strict=True)
        )

        precision, recall, f1 = compute_metrics(found_pairs, ground_truth)

        results.append(
            BenchmarkResult(
                strategy=f"threshold:{threshold}",
                precision=precision,
                recall=recall,
                f1=f1,
                runtime_ms=runtime_ms,
                n_matches=len(found_pairs),
            )
        )

    return results


def print_results(title: str, results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print("=" * 70)
    print(
        f"{'Strategy':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time(ms)':>10}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r.strategy:<25} {r.precision:>10.1%} {r.recall:>10.1%} "
            f"{r.f1:>10.1%} {r.runtime_ms:>10.1f}"
        )


def main() -> None:
    """Run all benchmarks."""
    print("Generating benchmark data...")
    source_df, target_df, ground_truth = generate_benchmark_data(
        n_records=500,
        n_duplicates=100,
        noise_level=0.3,
        seed=42,
    )

    print(f"Source records: {len(source_df)}")
    print(f"Target records: {len(target_df)}")
    print(f"True matches: {len(ground_truth)}")

    args = (source_df, target_df, ground_truth)
    print_results("Decision Methods", benchmark_decision_methods(*args))
    print_results("String Algorithms", benchmark_string_algorithms(*args))
    print_results("TF-IDF vs Standard", benchmark_tfidf_vs_standard(*args))
    print_results("Blocking", benchmark_blocking(*args))
    print_results("Threshold Sweep", benchmark_thresholds(*args))

    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print("""
Key findings from preclink benchmarks:

1. DECISION METHODS
   - Hungarian (Jonker-Volgenant): Optimal 1:1 matching, best for unique entities
   - Greedy: Fast but may miss optimal matches
   - Row-sequential: Good for streaming scenarios

2. STRING ALGORITHMS
   - Jaro-Winkler: Best for names (prefix-weighted)
   - Levenshtein: Good for typos/OCR errors
   - Damerau-Levenshtein: Handles transpositions well

3. TF-IDF WEIGHTING
   - Reduces false positives from common names (e.g., "John Smith")
   - Boosts rare name matches as stronger evidence
   - Recommended for real-world datasets with skewed name distributions

4. BLOCKING
   - Dramatically reduces runtime for large datasets
   - Trade-off: May miss matches across block boundaries
   - Use multiple blocking passes for better recall

5. THRESHOLD SELECTION
   - Higher threshold = higher precision, lower recall
   - Use F1 score to find optimal trade-off
   - Consider domain requirements (false positives vs false negatives)
""")


if __name__ == "__main__":
    main()
