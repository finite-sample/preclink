"""Type definitions for suture."""

from typing import Literal

DecisionMethod = Literal["hungarian", "greedy", "row_sequential"]
StringAlgorithm = Literal["jaro_winkler", "levenshtein", "damerau_levenshtein"]
MissingPolicy = Literal["skip", "zero", "penalize"]
