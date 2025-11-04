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
VARO - Variational Anti-Resonance Operator

The VARO operator implements a memory update mechanism that suppresses
resonant components to prevent the system from collapsing into a single
dominant representation. This enables the system to maintain cognitive
flexibility and explore novel conceptual spaces.

Paper reference: Equation 10 and Section 3.2 in "Aletheion: A Semi-Symbolic
Architecture for Internal Coherence Monitoring in Neural Language Systems"

Update formula:
    anti = z - β * dot(z, ψ_unit) * ψ_unit
    ψ_new = (1 - γ) * anti + γ * ψ_unit
    ψ_new = normalize(ψ_new)

Where:
- β controls anti-resonance strength (suppression of parallel component)
- γ controls memory persistence (EMA factor)
- ψ_unit is the normalized previous state
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

from .q_metric import normalize_vector


def varo_update(
    psi_prev: np.ndarray,
    z_new: np.ndarray,
    beta: float = 0.5,
    gamma: float = 0.9,
    eta: float = 0.0,
) -> np.ndarray:
    """
    Update symbolic state using the Variational Anti-Resonance Operator.

    This operator prevents cognitive collapse by suppressing the component
    of the new observation that resonates with the existing state, while
    maintaining memory continuity through exponential moving average.

    Parameters
    ----------
    psi_prev : np.ndarray
        Previous symbolic state (ψ_s at time t-1)
    z_new : np.ndarray
        New symbolic observation (z_out at time t)
    beta : float, default=0.5
        Anti-resonance strength in [0, 1]
        - β = 0: No suppression (standard EMA)
        - β = 1: Full suppression of parallel component
    gamma : float, default=0.9
        Memory persistence factor in [0, 1]
        - γ = 0: No memory (replace state completely)
        - γ = 1: Full memory (no update)
    eta : float, default=0.0
        Noise injection strength for exploration (typically 0 for basic use)

    Returns
    -------
    np.ndarray
        Updated symbolic state ψ_new (normalized to unit length)

    Examples
    --------
    >>> psi = np.array([1.0, 0.0, 0.0])
    >>> z = np.array([0.9, 0.1, 0.0])
    >>> psi_new = varo_update(psi, z, beta=0.5, gamma=0.9)
    >>> print(psi_new.shape)
    (3,)
    """
    # Ensure consistent dimensionality
    psi = np.asarray(psi_prev, dtype=np.float32).reshape(-1)
    z = np.asarray(z_new, dtype=np.float32).reshape(-1)

    if psi.size == 0 or z.size == 0:
        return np.array([], dtype=np.float32)

    # Normalize previous state
    psi_unit = normalize_vector(psi)

    # Compute anti-resonant component
    # This removes the part of z that aligns with the current state
    dot_product = float(np.dot(z, psi_unit)) if psi_unit.size > 0 else 0.0
    anti_resonant = z - beta * dot_product * psi_unit

    # Optional: Add exploratory noise
    if eta > 0:
        noise = np.random.normal(0.0, 1.0, size=anti_resonant.shape)
        anti_resonant = anti_resonant + eta * noise.astype(np.float32)

    # Blend anti-resonant component with previous state (EMA)
    new_state = (1.0 - gamma) * anti_resonant + gamma * psi_unit

    # Normalize to unit sphere
    return normalize_vector(new_state)


def lambda_mu_to_beta_gamma(lambda_ar: float, mu_ar: float) -> Tuple[float, float]:
    """
    Convert variational parameters (λ, μ) to implementation parameters (β, γ).

    This mapping allows theoretical control parameters to be translated into
    the practical update coefficients used by the VARO operator.

    Paper reference: Equation 10 in the Aletheion paper

    Mapping:
        (1 - γ) = 1 / (1 + μ)
        (1 - β) = 1 / (1 + λ + μ)

    Parameters
    ----------
    lambda_ar : float
        Anti-resonance intensity parameter (λ ≥ 0)
    mu_ar : float
        Memory persistence parameter (μ ≥ 0)

    Returns
    -------
    beta : float
        Anti-resonance strength in [0, 1)
    gamma : float
        Memory persistence factor in [0, 1)

    Examples
    --------
    >>> beta, gamma = lambda_mu_to_beta_gamma(lambda_ar=1.0, mu_ar=9.0)
    >>> print(f"β={beta:.3f}, γ={gamma:.3f}")
    β=0.909, γ=0.900
    """
    if lambda_ar < 0 or mu_ar < 0:
        raise ValueError("λ and μ must be non-negative")

    gamma = 1.0 - 1.0 / (1.0 + mu_ar)
    beta = 1.0 - 1.0 / (1.0 + lambda_ar + mu_ar)

    # Clamp to safe range
    beta = min(max(beta, 0.0), 0.999999)
    gamma = min(max(gamma, 0.0), 0.999999)

    return beta, gamma


__all__ = ["varo_update", "lambda_mu_to_beta_gamma", "normalize_vector"]
