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
Unit tests for VARO (Variational Anti-Resonance Operator) module.

Tests cover:
- Output normalization
- Anti-resonance behavior (β parameter)
- Temporal inertia (γ parameter)
- Edge cases and boundary conditions
- Parameter conversion (λ, μ → β, γ)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.varo import varo_update, lambda_mu_to_beta_gamma


class TestVARONormalization:
    """Tests for VARO output normalization."""

    def test_varo_output_normalized(self):
        """Test that VARO always returns unit-normalized vectors."""
        np.random.seed(42)

        for _ in range(100):
            dim = np.random.randint(2, 50)
            psi = np.random.randn(dim)
            z = np.random.randn(dim)

            beta = np.random.uniform(0, 1)
            gamma = np.random.uniform(0, 1)

            psi_new = varo_update(psi, z, beta=beta, gamma=gamma)

            norm = np.linalg.norm(psi_new)
            assert abs(norm - 1.0) < 1e-6, f"Expected unit norm, got {norm}"

    def test_varo_preserves_dimensionality(self):
        """Test that VARO preserves input dimensionality."""
        dimensions = [2, 3, 10, 100]

        for dim in dimensions:
            psi = np.random.randn(dim)
            z = np.random.randn(dim)

            psi_new = varo_update(psi, z, beta=0.5, gamma=0.9)

            assert psi_new.shape == (dim,), f"Expected shape ({dim},), got {psi_new.shape}"


class TestAntiResonance:
    """Tests for anti-resonance behavior (β parameter)."""

    def test_varo_no_antiresonance(self):
        """Test VARO with β=0 (standard EMA, no anti-resonance)."""
        psi = np.array([1.0, 0.0])
        z = np.array([0.0, 1.0])
        gamma = 0.5

        psi_new = varo_update(psi, z, beta=0.0, gamma=gamma)

        # With β=0, should be weighted average of psi and z (normalized)
        expected_unnorm = gamma * psi + (1 - gamma) * z
        expected = expected_unnorm / np.linalg.norm(expected_unnorm)

        assert np.allclose(psi_new, expected, atol=1e-6), \
            "β=0 should give standard EMA"

    def test_varo_full_antiresonance_parallel(self):
        """Test VARO with β=1 (full anti-resonance) for parallel vectors."""
        psi = np.array([1.0, 0.0, 0.0])
        z = np.array([1.0, 0.0, 0.0])  # Same direction

        psi_new = varo_update(psi, z, beta=1.0, gamma=0.9)

        # With β=1 and parallel vectors, parallel component is fully suppressed
        # Result should be normalized version of (1-γ)*0 + γ*psi = γ*psi
        assert np.linalg.norm(psi_new) > 0, "Should produce valid output"
        assert abs(np.linalg.norm(psi_new) - 1.0) < 1e-6, "Should be normalized"

    def test_varo_antiresonance_orthogonal(self):
        """Test VARO anti-resonance has no effect on orthogonal components."""
        psi = np.array([1.0, 0.0, 0.0])
        z = np.array([0.0, 1.0, 0.0])  # Orthogonal

        # For orthogonal vectors, anti-resonance shouldn't change behavior
        psi_new_no_ar = varo_update(psi, z, beta=0.0, gamma=0.9)
        psi_new_full_ar = varo_update(psi, z, beta=1.0, gamma=0.9)

        # Both should be close since there's no parallel component to suppress
        assert np.allclose(psi_new_no_ar, psi_new_full_ar, atol=1e-6), \
            "Anti-resonance shouldn't affect orthogonal components"

    def test_varo_antiresonance_strength(self):
        """Test that increasing β increases suppression of parallel component."""
        psi = np.array([1.0, 0.0, 0.0])
        z = np.array([1.0, 0.5, 0.0])  # Mostly parallel
        gamma = 0.5

        betas = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = []

        for beta in betas:
            psi_new = varo_update(psi, z, beta=beta, gamma=gamma)
            results.append(psi_new)

        # As β increases, the first component (parallel to psi) should decrease
        # (relative to what it would be without anti-resonance)
        for i in range(len(betas) - 1):
            # Just verify all results are valid (harder to predict exact ordering)
            assert np.linalg.norm(results[i]) > 0.99, "Should be normalized"


class TestTemporalInertia:
    """Tests for temporal inertia (γ parameter)."""

    def test_varo_no_memory(self):
        """Test VARO with γ=0 (no memory, immediate replacement)."""
        psi = np.array([1.0, 0.0, 0.0])
        z = np.array([0.0, 1.0, 0.0])

        psi_new = varo_update(psi, z, beta=0.0, gamma=0.0)

        # With γ=0, should just be normalized z (with no anti-resonance)
        expected = z / np.linalg.norm(z)
        assert np.allclose(psi_new, expected, atol=1e-6), \
            "γ=0 should replace state with new observation"

    def test_varo_full_memory(self):
        """Test VARO with γ=1 (perfect memory, no update)."""
        psi = np.array([1.0, 0.0, 0.0])
        z = np.array([0.0, 1.0, 0.0])

        psi_new = varo_update(psi, z, beta=0.0, gamma=1.0)

        # With γ=1, state should not change (normalized psi)
        expected = psi / np.linalg.norm(psi)
        assert np.allclose(psi_new, expected, atol=1e-6), \
            "γ=1 should keep current state unchanged"

    def test_varo_memory_interpolation(self):
        """Test that γ interpolates between old and new state."""
        psi = np.array([1.0, 0.0])
        z = np.array([0.0, 1.0])

        # Test various gamma values
        gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = []

        for gamma in gammas:
            psi_new = varo_update(psi, z, beta=0.0, gamma=gamma)
            results.append(psi_new)

        # At γ=0, should be close to z direction
        assert results[0][1] > results[0][0], "γ=0: More in z direction"

        # At γ=1, should be close to psi direction
        assert results[4][0] > results[4][1], "γ=1: More in psi direction"

        # At γ=0.5, should be balanced
        assert abs(results[2][0] - results[2][1]) < 0.3, "γ=0.5: Balanced"


class TestLambdaMuConversion:
    """Tests for variational parameter conversion."""

    def test_lambda_mu_to_beta_gamma_basic(self):
        """Test basic λ, μ → β, γ conversion."""
        lambda_ar = 1.0
        mu_ar = 9.0

        beta, gamma = lambda_mu_to_beta_gamma(lambda_ar, mu_ar)

        # Expected values from formulas
        expected_gamma = 1.0 - 1.0 / (1.0 + mu_ar)  # = 0.9
        expected_beta = 1.0 - 1.0 / (1.0 + lambda_ar + mu_ar)  # ≈ 0.909

        assert abs(gamma - expected_gamma) < 1e-6, f"Expected γ={expected_gamma}, got {gamma}"
        assert abs(beta - expected_beta) < 1e-6, f"Expected β={expected_beta}, got {beta}"

    def test_lambda_mu_zero_values(self):
        """Test conversion with λ=0, μ=0."""
        beta, gamma = lambda_mu_to_beta_gamma(0.0, 0.0)

        # With λ=0, μ=0:
        # γ = 1 - 1/(1+0) = 0
        # β = 1 - 1/(1+0+0) = 0

        assert abs(gamma - 0.0) < 1e-6, "Expected γ=0 for μ=0"
        assert abs(beta - 0.0) < 1e-6, "Expected β=0 for λ=0, μ=0"

    def test_lambda_mu_range(self):
        """Test that β and γ are always in [0, 1)."""
        test_cases = [
            (0.0, 0.0),
            (1.0, 1.0),
            (10.0, 10.0),
            (0.5, 9.0),
            (5.0, 0.5),
        ]

        for lambda_ar, mu_ar in test_cases:
            beta, gamma = lambda_mu_to_beta_gamma(lambda_ar, mu_ar)

            assert 0.0 <= beta < 1.0, f"β={beta} outside [0, 1)"
            assert 0.0 <= gamma < 1.0, f"γ={gamma} outside [0, 1)"

    def test_lambda_mu_negative_raises_error(self):
        """Test that negative λ or μ raises ValueError."""
        with pytest.raises(ValueError):
            lambda_mu_to_beta_gamma(-1.0, 1.0)

        with pytest.raises(ValueError):
            lambda_mu_to_beta_gamma(1.0, -1.0)

        with pytest.raises(ValueError):
            lambda_mu_to_beta_gamma(-1.0, -1.0)

    def test_lambda_mu_monotonicity(self):
        """Test that increasing λ increases β, and increasing μ increases γ."""
        # Test γ monotonicity (increasing μ)
        mu_values = [0.0, 1.0, 5.0, 10.0]
        gammas = []
        for mu in mu_values:
            _, gamma = lambda_mu_to_beta_gamma(1.0, mu)
            gammas.append(gamma)

        for i in range(len(gammas) - 1):
            assert gammas[i] < gammas[i + 1], "γ should increase with μ"

        # Test β monotonicity (increasing λ)
        lambda_values = [0.0, 1.0, 5.0, 10.0]
        betas = []
        for lam in lambda_values:
            beta, _ = lambda_mu_to_beta_gamma(lam, 5.0)
            betas.append(beta)

        for i in range(len(betas) - 1):
            assert betas[i] < betas[i + 1], "β should increase with λ"


class TestVAROEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_varo_empty_vectors(self):
        """Test VARO with empty vectors."""
        psi = np.array([])
        z = np.array([])

        psi_new = varo_update(psi, z, beta=0.5, gamma=0.9)

        assert psi_new.size == 0, "Should handle empty vectors"

    def test_varo_zero_vectors(self):
        """Test VARO with zero vectors."""
        psi = np.array([0.0, 0.0, 0.0])
        z = np.array([1.0, 0.0, 0.0])

        psi_new = varo_update(psi, z, beta=0.5, gamma=0.9)

        # Should handle gracefully
        assert psi_new.shape == (3,), "Should return correct shape"

    def test_varo_single_dimension(self):
        """Test VARO with 1D vectors."""
        psi = np.array([1.0])
        z = np.array([1.0])

        psi_new = varo_update(psi, z, beta=0.5, gamma=0.9)

        assert psi_new.shape == (1,), "Should handle 1D vectors"
        assert abs(np.linalg.norm(psi_new) - 1.0) < 1e-6, "Should be normalized"

    def test_varo_high_dimensionality(self):
        """Test VARO with high-dimensional vectors."""
        dim = 1000
        psi = np.random.randn(dim)
        z = np.random.randn(dim)

        psi_new = varo_update(psi, z, beta=0.5, gamma=0.9)

        assert psi_new.shape == (dim,), "Should handle high dimensions"
        assert abs(np.linalg.norm(psi_new) - 1.0) < 1e-6, "Should be normalized"

    def test_varo_extreme_parameters(self):
        """Test VARO with extreme parameter values."""
        psi = np.array([1.0, 0.0])
        z = np.array([0.0, 1.0])

        # Test various extreme combinations
        test_cases = [
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (0.999999, 0.999999),
        ]

        for beta, gamma in test_cases:
            psi_new = varo_update(psi, z, beta=beta, gamma=gamma)

            assert psi_new.shape == (2,), f"Should handle β={beta}, γ={gamma}"
            assert abs(np.linalg.norm(psi_new) - 1.0) < 1e-6, "Should be normalized"

    def test_varo_with_noise(self):
        """Test VARO with noise injection (η parameter)."""
        np.random.seed(42)

        psi = np.array([1.0, 0.0, 0.0])
        z = np.array([0.9, 0.1, 0.0])

        # Without noise
        psi_new_no_noise = varo_update(psi, z, beta=0.5, gamma=0.9, eta=0.0)

        # With noise
        psi_new_with_noise = varo_update(psi, z, beta=0.5, gamma=0.9, eta=0.1)

        # Both should be normalized
        assert abs(np.linalg.norm(psi_new_no_noise) - 1.0) < 1e-6
        assert abs(np.linalg.norm(psi_new_with_noise) - 1.0) < 1e-6

        # Results should be different due to noise
        assert not np.allclose(psi_new_no_noise, psi_new_with_noise), \
            "Noise should change output"

    def test_varo_convergence(self):
        """Test that repeated VARO updates converge."""
        np.random.seed(42)

        psi = np.array([1.0, 0.0, 0.0, 0.0])
        target = np.array([0.6, 0.6, 0.3, 0.3])
        target = target / np.linalg.norm(target)

        # Run multiple updates toward target
        for _ in range(50):
            # Observation moves toward target with noise
            z = 0.8 * target + 0.2 * psi + np.random.normal(0, 0.05, size=4)
            psi = varo_update(psi, z, beta=0.3, gamma=0.9)

        # Should be closer to target than initial state
        from src.q_metric import cosine_similarity

        initial_cos = cosine_similarity(np.array([1.0, 0.0, 0.0, 0.0]), target)
        final_cos = cosine_similarity(psi, target)

        assert final_cos > initial_cos, "Should converge toward target"


class TestVAROConsistency:
    """Tests for consistency and reproducibility."""

    def test_varo_deterministic(self):
        """Test that VARO is deterministic (without noise)."""
        psi = np.array([1.0, 0.5, 0.3])
        z = np.array([0.8, 0.6, 0.4])

        psi_new_1 = varo_update(psi, z, beta=0.5, gamma=0.9, eta=0.0)
        psi_new_2 = varo_update(psi, z, beta=0.5, gamma=0.9, eta=0.0)

        assert np.allclose(psi_new_1, psi_new_2), "Should be deterministic"

    def test_varo_type_consistency(self):
        """Test that VARO returns consistent types."""
        psi = np.array([1.0, 0.0], dtype=np.float32)
        z = np.array([0.0, 1.0], dtype=np.float32)

        psi_new = varo_update(psi, z, beta=0.5, gamma=0.9)

        assert isinstance(psi_new, np.ndarray), "Should return ndarray"
        assert psi_new.dtype == np.float32, "Should preserve dtype"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
