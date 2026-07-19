"""Unit tests for the ConPLex score-scale contract."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "inference"))

from conplex_scale import canonical_similarity  # noqa: E402


pytestmark = pytest.mark.unit


def test_native_similarity_is_not_affinely_rescaled():
    raw = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
    np.testing.assert_array_equal(canonical_similarity(raw), raw)


def test_roundoff_is_clipped_at_native_bounds():
    raw = np.array([-1e-8, 0.5, 1.0 + 1e-8])
    np.testing.assert_array_equal(
        canonical_similarity(raw), np.array([0.0, 0.5, 1.0])
    )


@pytest.mark.parametrize("raw", [
    np.array([-1e-3, 0.5]),
    np.array([0.5, 1.001]),
    np.array([0.5, np.nan]),
])
def test_invalid_scale_fails_explicitly(raw):
    with pytest.raises(ValueError):
        canonical_similarity(raw)


def test_native_threshold_and_similarity_share_the_same_scale():
    raw = np.array([0.49, 0.50, 0.51])
    threshold = 0.50
    scores = canonical_similarity(raw)
    np.testing.assert_array_equal(scores >= threshold, [False, True, True])
