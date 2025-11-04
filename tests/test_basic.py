# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025 Felipe M. Muniz
#
# This file is part of Aletheion Core (Educational Release).
#
# Aletheion Core is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# For commercial licensing inquiries, contact: licensing@aletheiaengine.dev

"""
Basic tests for Aletheion Core components

Run with: python -m pytest tests/test_basic.py
Or simply: python tests/test_basic.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import (
    check_quality_threshold,
    cosine_similarity,
    epistemic_gate,
    normalize_vector,
    q_metric,
    varo_update,
)


def test_q_metric_perfect_alignment():
    """Test Q metric for perfectly aligned vectors."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    q = q_metric(a, b)
    assert abs(q - 1.0) < 1e-6, f"Expected Q=1.0 for identical vectors, got {q}"
    print("✓ Q metric: perfect alignment")


def test_q_metric_orthogonal():
    """Test Q metric for orthogonal vectors."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    q = q_metric(a, b)
    assert abs(q - 0.5) < 1e-6, f"Expected Q=0.5 for orthogonal vectors, got {q}"
    print("✓ Q metric: orthogonal vectors")


def test_q_metric_opposite():
    """Test Q metric for opposite vectors."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    q = q_metric(a, b)
    assert abs(q - 0.0) < 1e-6, f"Expected Q=0.0 for opposite vectors, got {q}"
    print("✓ Q metric: opposite vectors")


def test_cosine_similarity():
    """Test cosine similarity computation."""
    a = np.array([1.0, 1.0])
    b = np.array([1.0, 0.0])
    cos = cosine_similarity(a, b)
    expected = np.sqrt(2) / 2  # cos(45°) ≈ 0.707
    assert abs(cos - expected) < 1e-6, f"Expected cos≈{expected}, got {cos}"
    print("✓ Cosine similarity")


def test_normalize_vector():
    """Test vector normalization."""
    v = np.array([3.0, 4.0])
    v_norm = normalize_vector(v)
    norm = np.linalg.norm(v_norm)
    assert abs(norm - 1.0) < 1e-6, f"Expected unit norm, got {norm}"
    assert abs(v_norm[0] - 0.6) < 1e-6, "Expected first component = 0.6"
    assert abs(v_norm[1] - 0.8) < 1e-6, "Expected second component = 0.8"
    print("✓ Vector normalization")


def test_varo_no_antiresonance():
    """Test VARO with β=0 (standard EMA)."""
    psi = np.array([1.0, 0.0])
    z = np.array([0.0, 1.0])
    psi_new = varo_update(psi, z, beta=0.0, gamma=0.5)
    # Should be weighted average, then normalized
    assert psi_new.shape == psi.shape
    assert abs(np.linalg.norm(psi_new) - 1.0) < 1e-6, "Result should be normalized"
    print("✓ VARO: no anti-resonance (β=0)")


def test_varo_with_antiresonance():
    """Test VARO with β>0 (anti-resonance active)."""
    psi = np.array([1.0, 0.0, 0.0])
    z = np.array([1.0, 0.0, 0.0])  # Same direction as psi
    psi_new = varo_update(psi, z, beta=1.0, gamma=0.5)

    # With full anti-resonance (β=1), parallel component should be suppressed
    assert psi_new.shape == psi.shape
    assert abs(np.linalg.norm(psi_new) - 1.0) < 1e-6
    print("✓ VARO: with anti-resonance (β=1)")


def test_quality_threshold_pass():
    """Test quality threshold check when gate should be open."""
    psi = np.array([1.0, 0.0])
    target = np.array([0.9, 0.1])
    gate_open, q = check_quality_threshold(psi, target, q_min=0.5)
    assert gate_open, "Gate should be open for high quality"
    assert q > 0.5, f"Quality {q} should exceed threshold 0.5"
    print("✓ Quality threshold: gate open")


def test_quality_threshold_fail():
    """Test quality threshold check when gate should be closed."""
    psi = np.array([1.0, 0.0])
    target = np.array([-1.0, 0.0])  # Opposite direction
    gate_open, q = check_quality_threshold(psi, target, q_min=0.5)
    assert not gate_open, "Gate should be closed for low quality"
    assert q < 0.5, f"Quality {q} should be below threshold 0.5"
    print("✓ Quality threshold: gate closed")


def test_epistemic_gate_accepts_good_state():
    """Test epistemic gate accepting high-quality state."""
    current = np.array([1.0, 0.0])
    proposed = np.array([0.95, 0.05])
    target = np.array([1.0, 0.0])

    accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.5)

    assert n_bt == 0, "Should not backtrack for good state"
    assert q >= 0.5, "Quality should meet threshold"
    # Accepted should be close to proposed
    assert np.allclose(
        accepted, proposed / np.linalg.norm(proposed)
    ), "Should accept proposed state"
    print("✓ Epistemic gate: accepts good state")


def test_epistemic_gate_backtracks_bad_state():
    """Test epistemic gate backtracking on low-quality state."""
    current = np.array([1.0, 0.0])
    proposed = np.array([0.0, 1.0])  # Orthogonal
    target = np.array([1.0, 0.0])

    accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.8)

    assert n_bt > 0, "Should backtrack for low quality state"
    assert q >= 0.8, "Final quality should meet threshold after backtracking"
    # Accepted should be between current and proposed
    print(f"✓ Epistemic gate: backtracks bad state ({n_bt} backtracks)")


def run_all_tests():
    """Run all test functions."""
    tests = [
        test_q_metric_perfect_alignment,
        test_q_metric_orthogonal,
        test_q_metric_opposite,
        test_cosine_similarity,
        test_normalize_vector,
        test_varo_no_antiresonance,
        test_varo_with_antiresonance,
        test_quality_threshold_pass,
        test_quality_threshold_fail,
        test_epistemic_gate_accepts_good_state,
        test_epistemic_gate_backtracks_bad_state,
    ]

    print("\n" + "=" * 70)
    print("  Running Aletheion Core Tests")
    print("=" * 70 + "\n")

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: Unexpected error: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
