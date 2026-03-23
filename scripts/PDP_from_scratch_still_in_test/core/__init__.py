"""
Core PDP analysis modules.
"""

from .pdp import (
    PDPAnalyzer,
    compute_inequality_matrix,
    compute_distance_matrix,
    PDPResult
)

__all__ = [
    "PDPAnalyzer",
    "compute_inequality_matrix", 
    "compute_distance_matrix",
    "PDPResult"
]
