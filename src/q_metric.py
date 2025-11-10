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
Q Metric - Epistemic Quality Measure

The Q metric quantifies epistemic coherence between symbolic representations.
It maps cosine similarity to a [0, 1] interval where:
- Q = 1.0: Perfect alignment (truth)
- Q = 0.5: Orthogonal (neutral)
- Q = 0.0: Complete opposition (falsehood)

Paper reference: Equation 1 in "Aletheion: A Semi-Symbolic Architecture for
Internal Coherence Monitoring in Neural Language Systems"

Formula: Q(ψ_s, ψ_t) = (1 + cos(ψ_s, ψ_t)) / 2
"""

from __future__ import annotations

from typing import Union

import numpy as np


def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    vector : np.ndarray
        Input vector to normalize
    eps : float
        Small constant to prevent division by zero

    Returns
    -------
    np.ndarray
        Unit-normalized vector
    """
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    a, b : np.ndarray
        Input vectors

    Returns
    -------
    float
        Cosine similarity in [-1, 1]
    """
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    similarity = float(np.dot(a_norm, b_norm))
    return max(-1.0, min(1.0, similarity))


def q_metric(psi_s: np.ndarray, psi_t: np.ndarray) -> float:
    """
    Compute the Q metric between two symbolic states.

    The Q metric transforms cosine similarity into an epistemic quality
    measure in [0, 1], where higher values indicate greater coherence
    between the symbolic representation (psi_s) and the target truth (psi_t).

    Parameters
    ----------
    psi_s : np.ndarray
        Source symbolic state (system representation)
    psi_t : np.ndarray
        Target symbolic state (ground truth or goal)

    Returns
    -------
    float
        Quality metric Q in [0, 1]

    Examples
    --------
    >>> psi_s = np.array([1.0, 0.0, 0.0])
    >>> psi_t = np.array([1.0, 0.0, 0.0])
    >>> q_metric(psi_s, psi_t)
    1.0

    >>> psi_s = np.array([1.0, 0.0, 0.0])
    >>> psi_t = np.array([0.0, 1.0, 0.0])
    >>> q_metric(psi_s, psi_t)
    0.5

    >>> psi_s = np.array([1.0, 0.0, 0.0])
    >>> psi_t = np.array([-1.0, 0.0, 0.0])
    >>> q_metric(psi_s, psi_t)
    0.0
    """
    cos_sim = cosine_similarity(psi_s, psi_t)
    return 0.5 * (1.0 + cos_sim)


__all__ = ["q_metric", "cosine_similarity", "normalize_vector"]
