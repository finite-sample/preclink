"""Compare dense and component-wise assignment on scored candidate pairs.

Run on macOS or Linux with:
    uv run --with recordlinkage python -m examples.benchmark_assignment

For observed bibliographic records, download the DBLP-ACM feature vector ZIP
from https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/ and run
    uv run python -m examples.benchmark_assignment --dblp-acm feature_vector.zip

Each CSV row measures decision time on a prepared candidate table in a fresh
process. Peak RSS is the process high-water mark, including imports and input.
Largest matrix size is a separate calculation from the candidate graph.
Febrl records are synthetic. Blocking and scoring use the existing Febrl benchmark.
"""

import argparse
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from preclink import HungarianDecision


def component_labels(pairs):
    left, left_ids = pd.factorize(pairs.left_index)
    right, right_ids = pd.factorize(pairs.right_index)
    size = len(left_ids) + len(right_ids)
    graph = coo_matrix(
        (np.ones(len(left), dtype=bool), (left, right + len(left_ids))),
        shape=(size, size),
    )
    _, labels = connected_components(graph, directed=False)
    return labels[left]


def decide_components(pairs):
    decide = HungarianDecision().decide
    if pairs.empty:
        return decide(pairs)
    labels = component_labels(pairs)
    if (labels == labels[0]).all():
        return decide(pairs)
    return pd.concat([decide(group) for _, group in pairs.groupby(labels, sort=False)])


def measure(method, pairs):
    decide = HungarianDecision().decide if method == "dense" else decide_components
    start = perf_counter()
    matches = decide(pairs)
    elapsed_ms = 1000 * (perf_counter() - start)
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    rss_mib = getrusage(RUSAGE_SELF).ru_maxrss / divisor
    if not (matches.left_index.is_unique and matches.right_index.is_unique):
        raise ValueError("A record was selected more than once")
    edges = list(zip(matches.left_index, matches.right_index, strict=True))
    return elapsed_ms, rss_mib, math.fsum(matches.score), edges


def benchmark(name, pairs, out):
    groups = pairs.groupby(component_labels(pairs))
    sizes = groups[["left_index", "right_index"]].nunique()
    dense_cells = pairs.left_index.nunique() * pairs.right_index.nunique()
    component_cells = sizes.prod(axis=1).max()
    for rep in range(3):
        order = ("dense", "components") if rep % 2 == 0 else ("components", "dense")
        results = {}
        for method in order:
            # Fresh interpreters keep the previous allocation out of peak RSS.
            with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as worker:
                results[method] = worker.submit(measure, method, pairs).result()
        _, _, dense_score, dense_edges = results["dense"]
        _, _, split_score, split_edges = results["components"]
        if len(dense_edges) != len(split_edges) or not math.isclose(
            dense_score, split_score, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{name} has different cardinality or objective")
        for method in order:
            ms, rss, score, edges = results[method]
            cells = dense_cells if method == "dense" else component_cells
            out.writerow(
                [
                    name,
                    rep + 1,
                    method,
                    len(pairs),
                    len(sizes),
                    cells * 8,
                    ms,
                    rss,
                    len(edges),
                    score,
                    len(set(edges) - set(dense_edges)),
                    edges == dense_edges,
                ]
            )
        sys.stdout.flush()


def febrl_pairs():
    # Keep dataset preparation imports out of the fresh decision workers.
    from recordlinkage.datasets import (
        load_febrl1,
        load_febrl2,
        load_febrl3,
        load_febrl4,
    )

    from examples.benchmark_febrl import split_febrl_data, standard_pipeline

    for i, load in enumerate((load_febrl1, load_febrl2, load_febrl3, load_febrl4), 1):
        if i == 4:
            left, right = load()
        else:
            left, right, _ = split_febrl_data(load())
        left["_id"] = left.index.astype(str)
        right["_id"] = right.index.astype(str)
        pairs = standard_pipeline().link(left, right).filtered_pairs
        yield f"febrl{i}", pairs


def dblp_acm_pairs(path):
    # Use the published all-field similarity for every candidate, independent
    # of the truth labels and the authors' train/validation/test partition.
    cols = {
        "source_id": "left_index",
        "target_id": "right_index",
        "cosine_tfidf": "score",
    }
    with (
        ZipFile(path) as archive,
        archive.open("feature_vector/feature_vector.csv") as f,
    ):
        return pd.read_csv(f, usecols=cols.keys(), dtype={"target_id": str}).rename(
            columns=cols
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dblp-acm", type=Path, metavar="FEATURE_VECTOR_ZIP")
    args = parser.parse_args()
    out = csv.writer(sys.stdout)
    out.writerow(
        [
            "dataset",
            "repetition",
            "method",
            "candidates",
            "components",
            "largest_matrix_bytes",
            "decision_ms",
            "peak_rss_mib",
            "matches",
            "total_score",
            "changed_pairs",
            "same_order",
        ]
    )
    if args.dblp_acm:
        benchmark("dblp-acm", dblp_acm_pairs(args.dblp_acm), out)
    else:
        for name, pairs in febrl_pairs():
            benchmark(name, pairs, out)


if __name__ == "__main__":
    main()
