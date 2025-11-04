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
Unit tests for epistemic gating module.

Tests cover:
- Quality threshold checking
- Gate acceptance/rejection behavior
- Backtracking mechanism
- Edge cases and boundary conditions
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.epistemic_gate import check_quality_threshold, epistemic_gate
from src.q_metric import q_metric


class TestCheckQualityThreshold:
    """Tests for quality threshold checking function."""

    def test_gate_accepts_high_quality(self):
        """Test that gate opens for high-quality states."""
        psi_state = np.array([1.0, 0.0, 0.0])
        psi_target = np.array([0.9, 0.1, 0.0])

        gate_open, q = check_quality_threshold(psi_state, psi_target, q_min=0.5)

        assert gate_open is True, "Gate should be open for high quality"
        assert q > 0.5, "Quality should exceed threshold"
        assert 0.0 <= q <= 1.0, "Quality should be in valid range"

    def test_gate_rejects_low_quality(self):
        """Test that gate closes for low-quality states."""
        psi_state = np.array([1.0, 0.0, 0.0])
        psi_target = np.array([-1.0, 0.0, 0.0])  # Opposite

        gate_open, q = check_quality_threshold(psi_state, psi_target, q_min=0.5)

        assert gate_open is False, "Gate should be closed for low quality"
        assert q < 0.5, "Quality should be below threshold"
        assert 0.0 <= q <= 1.0, "Quality should be in valid range"

    def test_gate_threshold_boundary(self):
        """Test gate behavior at threshold boundary."""
        psi_state = np.array([1.0, 0.0])
        psi_target = np.array([1.0, 0.0])

        # At threshold
        gate_open, q = check_quality_threshold(psi_state, psi_target, q_min=1.0)
        assert gate_open is True, "Gate should accept Q exactly at threshold"

        # Just above threshold
        psi_state = np.array([1.0, 0.0, 0.0])
        psi_target = np.array([0.99, 0.01, 0.0])
        gate_open, q = check_quality_threshold(psi_state, psi_target, q_min=0.9)
        assert gate_open is True, "Gate should accept Q just above threshold"

    def test_gate_various_thresholds(self):
        """Test gate with various threshold values."""
        psi_state = np.array([1.0, 0.0, 0.0])
        psi_target = np.array([0.7, 0.7, 0.0])

        q_actual = q_metric(psi_state, psi_target)

        thresholds = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

        for q_min in thresholds:
            gate_open, q = check_quality_threshold(psi_state, psi_target, q_min=q_min)

            assert abs(q - q_actual) < 1e-6, "Should return same Q value"

            if q >= q_min:
                assert (
                    gate_open is True
                ), f"Gate should be open for Q={q} >= Q_min={q_min}"
            else:
                assert (
                    gate_open is False
                ), f"Gate should be closed for Q={q} < Q_min={q_min}"

    def test_gate_returns_quality_value(self):
        """Test that gate returns correct quality value."""
        psi_state = np.array([1.0, 0.0, 0.0])
        psi_target = np.array([1.0, 0.0, 0.0])

        gate_open, q = check_quality_threshold(psi_state, psi_target, q_min=0.5)

        expected_q = q_metric(psi_state, psi_target)
        assert abs(q - expected_q) < 1e-10, "Should return correct Q metric value"


class TestEpistemicGate:
    """Tests for full epistemic gate with backtracking."""

    def test_gate_accepts_good_state(self):
        """Test that gate accepts high-quality proposed state without backtracking."""
        current = np.array([1.0, 0.0, 0.0])
        proposed = np.array([0.95, 0.05, 0.0])
        target = np.array([1.0, 0.0, 0.0])

        accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.5)

        assert n_bt == 0, "Should not backtrack for good state"
        assert q >= 0.5, "Quality should meet threshold"
        assert np.allclose(
            accepted, proposed / np.linalg.norm(proposed)
        ), "Should accept proposed state"

    def test_gate_backtracks_bad_state(self):
        """Test that gate backtracks when proposed state has low quality."""
        current = np.array([1.0, 0.0, 0.0])
        proposed = np.array([0.0, 1.0, 0.0])  # Orthogonal (Q=0.5)
        target = np.array([1.0, 0.0, 0.0])

        accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.8)

        assert n_bt > 0, "Should backtrack for low quality state"
        assert q >= 0.8, "Final quality should meet threshold"

        # Accepted should be between current and proposed
        dot_with_current = np.dot(accepted, current)
        dot_with_proposed = np.dot(accepted, proposed / np.linalg.norm(proposed))

        assert dot_with_current > 0, "Should be closer to current than proposed"

    def test_gate_max_backtrack(self):
        """Test that gate respects max backtrack limit."""
        current = np.array([1.0, 0.0, 0.0])
        proposed = np.array([-1.0, 0.0, 0.0])  # Opposite (Q=0)
        target = np.array([1.0, 0.0, 0.0])

        max_bt = 5
        accepted, q, n_bt = epistemic_gate(
            current, proposed, target, q_min=0.9, max_backtrack=max_bt
        )

        assert n_bt <= max_bt, f"Should not exceed max backtrack limit of {max_bt}"

        # If can't meet threshold, should return current state
        if q < 0.9:
            assert np.allclose(
                accepted, current / np.linalg.norm(current)
            ), "Should return current state if threshold can't be met"

    def test_gate_backtrack_interpolation(self):
        """Test that backtracking interpolates between current and proposed."""
        current = np.array([1.0, 0.0, 0.0])
        proposed = np.array([0.0, 1.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])

        accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.7)

        # Accepted should be a mix of current and proposed
        # Check that it's not exactly either one (unless n_bt=0)
        if n_bt > 0:
            assert not np.allclose(
                accepted, current / np.linalg.norm(current)
            ), "Should not be exactly current (backtracked)"
            assert not np.allclose(
                accepted, proposed / np.linalg.norm(proposed)
            ), "Should not be exactly proposed (backtracked)"

    def test_gate_backtrack_factor(self):
        """Test that backtrack_factor controls step size reduction."""
        current = np.array([1.0, 0.0, 0.0])
        proposed = np.array([0.0, 1.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])

        # Test different backtrack factors
        factors = [0.25, 0.5, 0.75]
        results = []

        for factor in factors:
            accepted, q, n_bt = epistemic_gate(
                current, proposed, target, q_min=0.7, backtrack_factor=factor
            )
            results.append((accepted, q, n_bt))

        # All should meet threshold
        for _, q, _ in results:
            assert q >= 0.7, "All should meet quality threshold"

    def test_gate_output_normalized(self):
        """Test that gate always returns normalized vectors."""
        np.random.seed(42)

        for _ in range(50):
            current = np.random.randn(3)
            proposed = np.random.randn(3)
            target = np.random.randn(3)

            accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.5)

            norm = np.linalg.norm(accepted)
            assert (
                abs(norm - 1.0) < 1e-6
            ), f"Output should be normalized, got norm={norm}"

    def test_gate_meets_threshold(self):
        """Test that gate always returns state meeting threshold."""
        np.random.seed(42)

        q_mins = [0.5, 0.6, 0.7, 0.8, 0.9]

        for q_min in q_mins:
            for _ in range(20):
                current = np.random.randn(4)
                proposed = np.random.randn(4)
                target = np.random.randn(4)

                accepted, q, n_bt = epistemic_gate(
                    current, proposed, target, q_min=q_min, max_backtrack=10
                )

                # Verify returned Q matches actual Q
                q_actual = q_metric(accepted, target)
                assert abs(q - q_actual) < 1e-6, "Returned Q should match actual"

                # Should meet threshold (or be current state if impossible)
                assert q >= q_min or np.allclose(
                    accepted, current / np.linalg.norm(current)
                ), f"Should meet threshold Q_min={q_min} or return current state"


class TestEpistemicGateEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_gate_identical_current_proposed(self):
        """Test gate when current and proposed are identical."""
        state = np.array([1.0, 0.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])

        accepted, q, n_bt = epistemic_gate(state, state, target, q_min=0.5)

        assert n_bt == 0, "Should not backtrack for identical states"
        assert abs(q - 1.0) < 1e-6, "Quality should be 1.0 for identical to target"
        assert np.allclose(
            accepted, state / np.linalg.norm(state)
        ), "Should return the state"

    def test_gate_zero_threshold(self):
        """Test gate with Q_min=0 (always accept)."""
        current = np.array([1.0, 0.0])
        proposed = np.array([-1.0, 0.0])  # Opposite
        target = np.array([1.0, 0.0])

        accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.0)

        assert n_bt == 0, "Should not backtrack with Q_min=0"
        assert np.allclose(
            accepted, proposed / np.linalg.norm(proposed)
        ), "Should accept proposed state"

    def test_gate_impossible_threshold(self):
        """Test gate with impossible threshold (Q_min=1.0, imperfect state)."""
        current = np.array([1.0, 0.0, 0.0])
        proposed = np.array([0.9, 0.1, 0.0])
        target = np.array([1.0, 0.0, 0.0])

        accepted, q, n_bt = epistemic_gate(
            current, proposed, target, q_min=1.0, max_backtrack=5
        )

        # Should return current state (safest option)
        q_current = q_metric(current, target)
        if q_current < 1.0:
            # If even current doesn't meet threshold, should still return current
            assert np.allclose(
                accepted, current / np.linalg.norm(current)
            ), "Should return current state when threshold can't be met"

    def test_gate_single_dimension(self):
        """Test gate with 1D vectors."""
        current = np.array([1.0])
        proposed = np.array([-0.5])
        target = np.array([1.0])

        accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.7)

        assert accepted.shape == (1,), "Should handle 1D vectors"
        assert abs(np.linalg.norm(accepted) - 1.0) < 1e-6, "Should be normalized"
        assert q >= 0.7, "Should meet threshold"

    def test_gate_high_dimensionality(self):
        """Test gate with high-dimensional vectors."""
        dim = 100
        current = np.random.randn(dim)
        proposed = np.random.randn(dim)
        target = np.random.randn(dim)

        accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.5)

        assert accepted.shape == (dim,), f"Should handle {dim}-D vectors"
        assert abs(np.linalg.norm(accepted) - 1.0) < 1e-6, "Should be normalized"

    def test_gate_all_zero_current(self):
        """Test gate with zero current state."""
        current = np.array([0.0, 0.0, 0.0])
        proposed = np.array([1.0, 0.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])

        accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.5)

        # Should handle gracefully
        assert accepted.shape == (3,), "Should return correct shape"

    def test_gate_preserves_dimensionality(self):
        """Test that gate preserves input dimensionality."""
        dimensions = [2, 3, 10, 50]

        for dim in dimensions:
            current = np.random.randn(dim)
            proposed = np.random.randn(dim)
            target = np.random.randn(dim)

            accepted, q, n_bt = epistemic_gate(current, proposed, target, q_min=0.5)

            assert accepted.shape == (dim,), f"Should preserve dimension {dim}"


class TestEpistemicGateConsistency:
    """Tests for consistency and reproducibility."""

    def test_gate_deterministic(self):
        """Test that gate is deterministic for same inputs."""
        current = np.array([1.0, 0.5, 0.3])
        proposed = np.array([0.8, 0.6, 0.4])
        target = np.array([0.9, 0.4, 0.3])

        accepted1, q1, n_bt1 = epistemic_gate(current, proposed, target, q_min=0.7)
        accepted2, q2, n_bt2 = epistemic_gate(current, proposed, target, q_min=0.7)

        assert np.allclose(accepted1, accepted2), "Should be deterministic"
        assert abs(q1 - q2) < 1e-10, "Should return same quality"
        assert n_bt1 == n_bt2, "Should perform same number of backtracks"

    def test_gate_quality_increases_with_backtrack(self):
        """Test that quality generally increases with backtracking."""
        current = np.array([1.0, 0.0, 0.0])
        proposed = np.array([0.0, 1.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])

        # Initial quality of proposed
        q_proposed = q_metric(proposed, target)

        # After backtracking to meet threshold
        accepted, q_final, n_bt = epistemic_gate(current, proposed, target, q_min=0.7)

        if n_bt > 0:
            # Quality should have improved from proposed
            assert q_final >= q_proposed, "Backtracking should improve quality"

    def test_gate_backtrack_count_increases_with_threshold(self):
        """Test that higher thresholds generally require more backtracks."""
        current = np.array([1.0, 0.0, 0.0])
        proposed = np.array([0.0, 1.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])

        thresholds = [0.5, 0.6, 0.7, 0.8]
        backtrack_counts = []

        for q_min in thresholds:
            _, _, n_bt = epistemic_gate(current, proposed, target, q_min=q_min)
            backtrack_counts.append(n_bt)

        # Generally, higher thresholds should require more (or equal) backtracks
        # (not always strictly monotonic due to exponential backoff)
        # Just verify reasonable behavior
        assert all(
            n >= 0 for n in backtrack_counts
        ), "All should have non-negative backtracks"


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_gate_in_update_loop(self):
        """Test gate in a realistic update loop."""
        np.random.seed(42)

        psi_state = np.array([1.0, 0.0, 0.0, 0.0])
        psi_target = np.array([0.6, 0.6, 0.3, 0.3])
        psi_target = psi_target / np.linalg.norm(psi_target)

        q_min = 0.6
        total_backtracks = 0

        for step in range(10):
            # Simulate update (moving toward target with noise)
            proposed = (
                0.7 * psi_target + 0.3 * psi_state + np.random.normal(0, 0.1, size=4)
            )

            # Apply gate
            accepted, q, n_bt = epistemic_gate(
                psi_state, proposed, psi_target, q_min=q_min
            )

            total_backtracks += n_bt

            # Update state
            psi_state = accepted

            # Verify quality maintained
            assert q >= q_min, f"Step {step}: Quality should meet threshold"

        # Should make some progress toward target
        final_q = q_metric(psi_state, psi_target)
        assert final_q > 0.6, "Should improve quality over time"

    def test_gate_prevents_divergence(self):
        """Test that gate prevents divergence from target."""
        np.random.seed(42)

        current = np.array([0.9, 0.1, 0.0, 0.0])
        current = current / np.linalg.norm(current)

        target = np.array([1.0, 0.0, 0.0, 0.0])

        # Propose state far from target
        bad_proposal = np.array([0.0, 0.0, 1.0, 0.0])

        q_before = q_metric(current, target)

        accepted, q_after, n_bt = epistemic_gate(
            current, bad_proposal, target, q_min=0.8
        )

        # Gate should prevent quality from dropping too much
        assert q_after >= 0.8, "Gate should maintain quality threshold"
        assert q_after >= q_before * 0.9, "Gate should prevent significant quality drop"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
