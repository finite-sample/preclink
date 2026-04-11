# preclink

[![PyPI version](https://badge.fury.io/py/preclink.svg)](https://badge.fury.io/py/preclink)
[![Downloads](https://pepy.tech/badge/preclink)](https://pepy.tech/project/preclink)
[![CI](https://github.com/finite-sample/preclink/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/preclink/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://finite-sample.github.io/preclink)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

High-precision record linkage library implementing a 7-step pipeline with multi-pass support.

## Installation

```bash
pip install preclink
```

## Quick Start

```python
import pandas as pd
from preclink import Pipeline, StringComparison, ExactComparison

df_left = pd.DataFrame({
    "id": [1, 2, 3],
    "first_name": ["John", "Jane", "Bob"],
    "last_name": ["Smith", "Doe", "Johnson"],
    "state": ["CA", "NY", "CA"],
})

df_right = pd.DataFrame({
    "id": [101, 102, 103],
    "first_name": ["Jon", "Jane", "Robert"],
    "last_name": ["Smith", "Doe", "Johnson"],
    "state": ["CA", "NY", "CA"],
})

result = (
    Pipeline()
    .preprocess(normalize_unicode=True, lowercase=True)
    .block(on="state")
    .score(comparisons=[
        StringComparison("first_name", algorithm="jaro_winkler"),
        StringComparison("last_name", algorithm="jaro_winkler"),
    ])
    .filter(min_score=0.7)
    .decide(method="hungarian")
    .build()
    .link(df_left, df_right)
)

print(result.matches)
```

## The 7-Step Pipeline

1. **Preprocess**: Normalize text (unicode, case, whitespace)
2. **Deduplicate**: Remove within-table duplicates
3. **Block**: Reduce comparison space using blocking keys
4. **Score**: Compute pairwise similarity scores
5. **Filter**: Remove low-confidence pairs
6. **Decide**: Apply matching algorithm (Hungarian, Greedy, Row-Sequential)
7. **Inspect**: Generate diagnostics and reports

## Multi-Pass Matching

For complex datasets, use multi-pass matching with progressively relaxed thresholds:

```python
from preclink import MultiPassOrchestrator, StringComparison

orchestrator = MultiPassOrchestrator()
result = orchestrator.run(
    df_left,
    df_right,
    passes=[
        {"min_score": 0.95, "method": "hungarian"},
        {"min_score": 0.85, "method": "hungarian"},
        {"min_score": 0.70, "method": "greedy"},
    ],
    comparisons=[
        StringComparison("first_name"),
        StringComparison("last_name"),
    ],
)
```

## Decision Rules

- **Hungarian**: Optimal assignment maximizing total score (recommended for precision)
- **Greedy**: Best-global-pair first, fast and precision-optimized
- **Row-Sequential**: Process left records in order (baseline)

## Features

- Type-safe with full mypy strict mode support
- Extensible via protocols (custom comparisons, blockers, decision rules)
- Native pandas DataFrames throughout
- Crosswalk support for blocking key normalization
- Margin-based filtering for ambiguity removal

## Benchmarks

Comparison against [recordlinkage](https://github.com/J535D165/recordlinkage) on standard Febrl datasets:

| Dataset | Library | Precision | Recall | F1 |
|---------|---------|-----------|--------|-----|
| febrl1 | preclink | **100.0%** | 76.4% | 86.6% |
| febrl1 | recordlinkage | 99.5% | 79.0% | 88.1% |
| febrl2 | preclink | **97.3%** | 39.5% | 56.2% |
| febrl2 | recordlinkage | 95.0% | 80.0% | 86.9% |
| febrl3 | preclink | **99.2%** | 35.4% | 52.2% |
| febrl3 | recordlinkage | 98.1% | 79.6% | 87.9% |
| febrl4 | preclink | **99.9%** | 79.0% | **88.2%** |
| febrl4 | recordlinkage | 94.3% | 80.8% | 87.0% |

**febrl4** is the true record linkage scenario (linking two separate tables). On this dataset, preclink achieves 99.9% precision with higher F1 than recordlinkage. The other datasets (febrl1-3) are deduplication tasks where records are split artificially.

**When to use preclink**: When false positives are costly (merging administrative records, survey linking, research applications) and you need provably optimal 1:1 matching.

Reproduce:
```bash
pip install recordlinkage
python examples/benchmark_febrl.py
```

## Documentation

Full documentation at [finite-sample.github.io/preclink](https://finite-sample.github.io/preclink)

## License

MIT License
