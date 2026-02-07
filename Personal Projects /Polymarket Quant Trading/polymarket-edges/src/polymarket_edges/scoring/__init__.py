"""Scoring module for v2 multi-component ranking.

This package contains the v2 scorer. For v1 compatibility,
import compute_scores from the parent scoring_v1 module.
"""

from polymarket_edges.scoring.scorer_v2 import ScorerV2

__all__ = ["ScorerV2"]
