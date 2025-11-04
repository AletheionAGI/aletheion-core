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
Unit tests for Q metric module.

Tests cover:
- Q metric range validation
- Special cases (identical, orthogonal, opposite vectors)
- Cosine similarity computation
- Vector normalization
- Edge cases (zero vectors, different dimensionalities)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.q_metric import cosine_similarity, normalize_vector, q_metric


class TestQMetric:
    """Tests for the Q metric function."""

    def test_q_metric_range(self):
        """Test that Q metric always returns values in [0, 1]."""
        np.random.seed(42)

        for _ in range(100):
            # Generate random vectors of various dimensions
            dim = np.random.randint(2, 100)
            a = np.random.randn(dim)
            b = np.random.randn(dim)

            q = q_metric(a, b)

            assert 0.0 <= q <= 1.0, f"Q={q} outside valid range [0, 1]"
            assert isinstance(q, float), "Q should be a float"

    def test_q_metric_identical_vectors(self):
        """Test Q metric returns 1.0 for identical vectors."""
        test_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([-1.0, 0.0, 0.0]),
        ]

        for vec in test_vectors:
            q = q_metric(vec, vec)
            assert abs(q - 1.0) < 1e-6, f"Expected Q=1.0 for identical vectors, got {q}"

    def test_q_metric_orthogonal_vectors(self):
        """Test Q metric returns 0.5 for orthogonal vectors."""
        test_cases = [
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([1.0, 1.0, 0.0]), np.array([1.0, -1.0, 0.0])),
            (np.array([2.0, 0.0]), np.array([0.0, 3.0])),
        ]

        for a, b in test_cases:
            q = q_metric(a, b)
            assert (
                abs(q - 0.5) < 1e-6
            ), f"Expected Q=0.5 for orthogonal vectors, got {q}"

    def test_q_metric_opposite_vectors(self):
        """Test Q metric returns ~0.0 for opposite vectors."""
        test_cases = [
            (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])),
            (np.array([1.0, 1.0]), np.array([-1.0, -1.0])),
            (np.array([3.0, 4.0]), np.array([-3.0, -4.0])),
        ]

        for a, b in test_cases:
            q = q_metric(a, b)
            assert abs(q - 0.0) < 1e-6, f"Expected Q≈0.0 for opposite vectors, got {q}"

    def test_q_metric_symmetry(self):
        """Test that Q metric is symmetric: Q(a,b) = Q(b,a)."""
        np.random.seed(42)

        for _ in range(50):
            dim = np.random.randint(2, 50)
            a = np.random.randn(dim)
            b = np.random.randn(dim)

            q_ab = q_metric(a, b)
            q_ba = q_metric(b, a)

            assert abs(q_ab - q_ba) < 1e-10, "Q metric should be symmetric"

    def test_q_metric_scale_invariance(self):
        """Test that Q metric is invariant to vector scaling."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        q_original = q_metric(a, b)

        # Scale vectors
        q_scaled_a = q_metric(10 * a, b)
        q_scaled_b = q_metric(a, 0.1 * b)
        q_scaled_both = q_metric(5 * a, 2 * b)

        # Use appropriate tolerance for float32 precision (~1e-7)
        assert abs(q_original - q_scaled_a) < 1e-6, "Q should be scale invariant"
        assert abs(q_original - q_scaled_b) < 1e-6, "Q should be scale invariant"
        assert abs(q_original - q_scaled_both) < 1e-6, "Q should be scale invariant"

    def test_q_metric_different_dimensions_handled(self):
        """Test that Q metric handles vectors that can be flattened."""
        # All should be treated as 3D vectors after flattening
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[1.0], [0.0], [0.0]])  # Different shape but same data

        q = q_metric(a, b.flatten())
        assert abs(q - 1.0) < 1e-6


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity is 1.0 for identical vectors."""
        vectors = [
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([3.0, 4.0]),
        ]

        for vec in vectors:
            cos_sim = cosine_similarity(vec, vec)
            assert abs(cos_sim - 1.0) < 1e-6, f"Expected cos=1.0, got {cos_sim}"

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity is 0.0 for orthogonal vectors."""
        test_cases = [
            (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            (np.array([2.0, 0.0]), np.array([0.0, 5.0])),
        ]

        for a, b in test_cases:
            cos_sim = cosine_similarity(a, b)
            assert abs(cos_sim - 0.0) < 1e-6, f"Expected cos=0.0, got {cos_sim}"

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity is -1.0 for opposite vectors."""
        test_cases = [
            (np.array([1.0, 0.0]), np.array([-1.0, 0.0])),
            (np.array([1.0, 1.0]), np.array([-1.0, -1.0])),
            (np.array([3.0, 4.0, 5.0]), np.array([-3.0, -4.0, -5.0])),
        ]

        for a, b in test_cases:
            cos_sim = cosine_similarity(a, b)
            assert abs(cos_sim - (-1.0)) < 1e-6, f"Expected cos=-1.0, got {cos_sim}"

    def test_cosine_similarity_45_degrees(self):
        """Test cosine similarity for 45-degree angle."""
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 0.0])

        cos_sim = cosine_similarity(a, b)
        expected = np.sqrt(2) / 2  # cos(45°) ≈ 0.707

        assert abs(cos_sim - expected) < 1e-6

    def test_cosine_similarity_range(self):
        """Test cosine similarity always returns values in [-1, 1]."""
        np.random.seed(42)

        for _ in range(100):
            dim = np.random.randint(2, 50)
            a = np.random.randn(dim)
            b = np.random.randn(dim)

            cos_sim = cosine_similarity(a, b)

            assert -1.0 <= cos_sim <= 1.0, f"cos_sim={cos_sim} outside [-1, 1]"


class TestNormalizeVector:
    """Tests for vector normalization function."""

    def test_normalize_vector_creates_unit_vector(self):
        """Test that normalized vectors have unit length."""
        test_vectors = [
            np.array([3.0, 4.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([5.0, -12.0]),
        ]

        for vec in test_vectors:
            normalized = normalize_vector(vec)
            norm = np.linalg.norm(normalized)

            assert abs(norm - 1.0) < 1e-6, f"Expected unit norm, got {norm}"

    def test_normalize_vector_preserves_direction(self):
        """Test that normalization preserves vector direction."""
        vec = np.array([3.0, 4.0])
        normalized = normalize_vector(vec)

        # Check that normalized vector is parallel to original
        cos_sim = cosine_similarity(vec, normalized)
        assert abs(cos_sim - 1.0) < 1e-6, "Normalization should preserve direction"

    def test_normalize_specific_values(self):
        """Test normalization with known values."""
        vec = np.array([3.0, 4.0])  # Norm = 5
        normalized = normalize_vector(vec)

        expected = np.array([0.6, 0.8])
        assert np.allclose(
            normalized, expected
        ), f"Expected {expected}, got {normalized}"

    def test_normalize_already_normalized(self):
        """Test that normalizing a unit vector returns the same vector."""
        vec = np.array([1.0, 0.0, 0.0])
        normalized = normalize_vector(vec)

        assert np.allclose(
            vec, normalized
        ), "Normalizing unit vector should be identity"

    def test_normalize_zero_vector(self):
        """Test that normalizing zero vector returns zero vector."""
        vec = np.array([0.0, 0.0, 0.0])
        normalized = normalize_vector(vec)

        expected = np.array([0.0, 0.0, 0.0])
        assert np.allclose(normalized, expected), "Zero vector should normalize to zero"

    def test_normalize_near_zero_vector(self):
        """Test that normalizing near-zero vector returns zero vector."""
        vec = np.array([1e-15, 1e-15, 1e-15])
        normalized = normalize_vector(vec)

        # Should be treated as zero due to eps threshold
        assert np.allclose(normalized, 0.0), "Near-zero vector should normalize to zero"

    def test_normalize_negative_values(self):
        """Test normalization with negative values."""
        vec = np.array([-3.0, -4.0])
        normalized = normalize_vector(vec)

        norm = np.linalg.norm(normalized)
        assert (
            abs(norm - 1.0) < 1e-6
        ), "Should create unit vector with negative components"

        expected = np.array([-0.6, -0.8])
        assert np.allclose(normalized, expected)

    def test_normalize_custom_eps(self):
        """Test normalize_vector with custom epsilon."""
        vec = np.array([1e-8, 1e-8])

        # With default eps (1e-12), should normalize
        normalized_default = normalize_vector(vec)
        assert not np.allclose(normalized_default, 0.0)

        # With larger eps, should return zero
        normalized_custom = normalize_vector(vec, eps=1e-6)
        assert np.allclose(normalized_custom, 0.0)


class TestQMetricEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_q_metric_zero_vectors(self):
        """Test Q metric with zero vectors."""
        zero = np.array([0.0, 0.0, 0.0])
        normal = np.array([1.0, 0.0, 0.0])

        q = q_metric(zero, normal)
        # Should return 0.5 (neutral) for zero vectors
        assert abs(q - 0.5) < 1e-6

        q_both_zero = q_metric(zero, zero)
        assert abs(q_both_zero - 0.5) < 1e-6

    def test_q_metric_single_dimension(self):
        """Test Q metric with 1D vectors."""
        a = np.array([1.0])
        b = np.array([1.0])

        q = q_metric(a, b)
        assert abs(q - 1.0) < 1e-6

        a = np.array([1.0])
        b = np.array([-1.0])

        q = q_metric(a, b)
        assert abs(q - 0.0) < 1e-6

    def test_q_metric_high_dimensionality(self):
        """Test Q metric with high-dimensional vectors."""
        dim = 1000
        a = np.random.randn(dim)
        b = np.random.randn(dim)

        q = q_metric(a, b)

        assert 0.0 <= q <= 1.0
        assert isinstance(q, float)

    def test_q_metric_very_small_values(self):
        """Test Q metric with very small vector components."""
        a = np.array([1e-10, 1e-10, 1e-10])
        b = np.array([1e-10, 1e-10, 1e-10])

        q = q_metric(a, b)
        # Should still recognize as identical direction
        assert (
            abs(q - 1.0) < 1e-6 or abs(q - 0.5) < 1e-6
        )  # Either identical or zero-like

    def test_q_metric_mixed_signs(self):
        """Test Q metric with mixed positive/negative values."""
        a = np.array([1.0, -1.0, 1.0])
        b = np.array([1.0, -1.0, 1.0])

        q = q_metric(a, b)
        assert (
            abs(q - 1.0) < 1e-6
        ), "Identical vectors should give Q=1 regardless of signs"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
