from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_ridge(alpha: float = 1.0):
    """Linear ridge. Cannot represent conditional (multiplicative) effects."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def make_interaction_ridge(alpha: float = 1.0):
    """Ridge over standardized features plus all pairwise products.

    Needed for modalities that matter only in combination with another, e.g.
    audio whose effect depends on pose.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("model", Ridge(alpha=alpha)),
    ])
