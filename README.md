# tether

[![PyPI version](https://badge.fury.io/py/tether.svg)](https://badge.fury.io/py/tether)
[![CI](https://github.com/finite-sample/tether/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/tether/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/tether/badge/?version=latest)](https://tether.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

High-precision record linkage library implementing a 7-step pipeline with multi-pass support.

## Installation

```bash
pip install tether
```

## Quick Start

```python
import pandas as pd
from tether import Pipeline, StringComparison, ExactComparison

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
from tether import MultiPassOrchestrator, StringComparison

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

## Documentation

Full documentation at [tether.readthedocs.io](https://tether.readthedocs.io)

## License

MIT License
