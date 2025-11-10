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
Epistemic Gating - Quality Threshold Enforcement

The epistemic gate ensures that updates to the symbolic state maintain
a minimum level of coherence with the target. When a proposed update
would reduce quality below the threshold Q_min, the gate "closes" and
the update is rejected or modulated.

This mechanism prevents the system from accepting low-quality representations
that would compromise epistemic integrity.

Paper reference: Section 4.3 in "Aletheion: A Semi-Symbolic Architecture for
Internal Coherence Monitoring in Neural Language Systems"
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .q_metric import q_metric


def check_quality_threshold(
    psi_state: np.ndarray,
    psi_target: np.ndarray,
    q_min: float = 0.5,
) -> Tuple[bool, float]:
    """
    Check if the symbolic state meets the minimum quality threshold.

    Parameters
    ----------
    psi_state : np.ndarray
        Current or proposed symbolic state
    psi_target : np.ndarray
        Target symbolic state (ground truth or goal)
    q_min : float, default=0.5
        Minimum acceptable quality threshold in [0, 1]

    Returns
    -------
    gate_open : bool
        True if quality >= q_min (gate is open, accept state)
        False if quality < q_min (gate is closed, reject state)
    quality : float
        The computed Q metric value

    Examples
    --------
    >>> psi_state = np.array([1.0, 0.0, 0.0])
    >>> psi_target = np.array([0.9, 0.1, 0.0])
    >>> gate_open, q = check_quality_threshold(psi_state, psi_target, q_min=0.5)
    >>> print(f"Gate open: {gate_open}, Q={q:.3f}")
    Gate open: True, Q=0.995
    """
    quality = q_metric(psi_state, psi_target)
    gate_open = quality >= q_min
    return gate_open, quality


def epistemic_gate(
    psi_current: np.ndarray,
    psi_proposed: np.ndarray,
    psi_target: np.ndarray,
    q_min: float = 0.5,
    max_backtrack: int = 8,
    backtrack_factor: float = 0.5,
) -> Tuple[np.ndarray, float, int]:
    """
    Apply epistemic gating with automatic backtracking.

    If the proposed state fails to meet the quality threshold, the gate
    automatically backtracks by interpolating between the current and
    proposed states until the threshold is met or max iterations reached.

    This implements the "exploratory step" mechanism described in the paper.

    Parameters
    ----------
    psi_current : np.ndarray
        Current symbolic state (safe baseline)
    psi_proposed : np.ndarray
        Proposed new symbolic state (to be validated)
    psi_target : np.ndarray
        Target symbolic state (ground truth or goal)
    q_min : float, default=0.5
        Minimum acceptable quality threshold
    max_backtrack : int, default=8
        Maximum number of backtracking iterations
    backtrack_factor : float, default=0.5
        Factor by which to reduce the step size on each backtrack

    Returns
    -------
    psi_accepted : np.ndarray
        Accepted state (either proposed or backtracked version)
    final_quality : float
        Quality of the accepted state
    num_backtracks : int
        Number of backtracking iterations performed

    Examples
    --------
    >>> current = np.array([1.0, 0.0, 0.0])
    >>> proposed = np.array([0.0, 1.0, 0.0])
    >>> target = np.array([1.0, 0.0, 0.0])
    >>> accepted, q, n = epistemic_gate(current, proposed, target, q_min=0.8)
    >>> print(f"Backtracks: {n}, Q={q:.3f}")
    Backtracks: 3, Q=0.854
    """
    psi_curr = np.asarray(psi_current, dtype=np.float32).reshape(-1)
    psi_prop = np.asarray(psi_proposed, dtype=np.float32).reshape(-1)

    # Normalize current state
    norm_curr = np.linalg.norm(psi_curr)
    if norm_curr > 1e-12:
        psi_curr = psi_curr / norm_curr

    # Normalize proposed state
    norm_prop = np.linalg.norm(psi_prop)
    if norm_prop > 1e-12:
        psi_prop = psi_prop / norm_prop

    # Get quality of current state to prevent excessive quality drops
    _, q_current = check_quality_threshold(psi_curr, psi_target, q_min)
    # Don't accept states that drop quality by more than 10% from current
    # Only enforce this when q_min is reasonably high (> 0.5)
    if q_min > 0.5:
        effective_q_min = max(q_min, q_current * 0.9)
    else:
        effective_q_min = q_min

    # Check if proposed state passes gate
    gate_open, quality = check_quality_threshold(psi_prop, psi_target, effective_q_min)

    if gate_open:
        # Gate is open, accept proposed state (already normalized)
        return psi_prop, quality, 0

    # Gate is closed, perform backtracking
    step_size = 1.0
    num_backtracks = 0
    psi_candidate = psi_prop.copy()

    for i in range(max_backtrack):
        step_size *= backtrack_factor
        num_backtracks += 1

        # Interpolate between current (safe) and proposed (risky)
        psi_candidate = (1.0 - step_size) * psi_curr + step_size * psi_prop

        # Renormalize
        norm = np.linalg.norm(psi_candidate)
        if norm > 1e-12:
            psi_candidate = psi_candidate / norm

        # Check if backtracked state passes gate
        gate_open, quality = check_quality_threshold(
            psi_candidate, psi_target, effective_q_min
        )

        if gate_open:
            return psi_candidate, quality, num_backtracks

    # Max backtracks reached, return current state (safest option)
    gate_open, quality = check_quality_threshold(psi_curr, psi_target, q_min)
    return psi_curr, quality, num_backtracks


__all__ = ["check_quality_threshold", "epistemic_gate"]
