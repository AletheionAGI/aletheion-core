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
Simple Demonstration of Aletheion Core

This demo shows how the three core components work together:
1. Q metric: Measures coherence between states
2. VARO operator: Updates state with anti-resonance
3. Epistemic gate: Enforces quality thresholds

Scenario: A symbolic AGI maintains coherence while learning from observations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src import q_metric, varo_update, epistemic_gate, check_quality_threshold


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def demo_q_metric():
    """Demonstrate the Q metric for measuring epistemic quality."""
    print_header("Demo 1: Q Metric - Epistemic Quality Measure")

    # Define some example states
    truth = np.array([1.0, 0.0, 0.0])
    aligned = np.array([0.9, 0.1, 0.0])
    orthogonal = np.array([0.0, 1.0, 0.0])
    opposite = np.array([-1.0, 0.0, 0.0])

    print("\nTruth vector: [1.0, 0.0, 0.0]")
    print("\nMeasuring coherence with different states:")
    print(f"  Aligned     [0.9, 0.1, 0.0]: Q = {q_metric(aligned, truth):.4f}")
    print(f"  Orthogonal  [0.0, 1.0, 0.0]: Q = {q_metric(orthogonal, truth):.4f}")
    print(f"  Opposite   [-1.0, 0.0, 0.0]: Q = {q_metric(opposite, truth):.4f}")

    print("\nInterpretation:")
    print("  Q ≈ 1.0 → Perfect alignment (truth)")
    print("  Q ≈ 0.5 → Orthogonal (neutral)")
    print("  Q ≈ 0.0 → Complete opposition (falsehood)")


def demo_varo():
    """Demonstrate the VARO operator for anti-resonance updates."""
    print_header("Demo 2: VARO - Anti-Resonance State Update")

    # Initial state: System believes in concept [1, 0, 0]
    psi_initial = np.array([1.0, 0.0, 0.0])

    # New observation: Slightly different [0.95, 0.05, 0]
    z_obs = np.array([0.95, 0.05, 0.0])

    print("\nInitial state ψ:     [1.00, 0.00, 0.00]")
    print("New observation z:   [0.95, 0.05, 0.00]")

    print("\n--- Standard EMA update (β=0, no anti-resonance) ---")
    psi_ema = varo_update(psi_initial, z_obs, beta=0.0, gamma=0.9)
    print(f"Result: [{psi_ema[0]:.3f}, {psi_ema[1]:.3f}, {psi_ema[2]:.3f}]")
    print("Effect: Converges toward observation (risk of collapse)")

    print("\n--- VARO update (β=0.5, moderate anti-resonance) ---")
    psi_varo = varo_update(psi_initial, z_obs, beta=0.5, gamma=0.9)
    print(f"Result: [{psi_varo[0]:.3f}, {psi_varo[1]:.3f}, {psi_varo[2]:.3f}]")
    print("Effect: Suppresses parallel component, maintains exploration")

    print("\n--- VARO update (β=1.0, full anti-resonance) ---")
    psi_full = varo_update(psi_initial, z_obs, beta=1.0, gamma=0.9)
    print(f"Result: [{psi_full[0]:.3f}, {psi_full[1]:.3f}, {psi_full[2]:.3f}]")
    print("Effect: Maximum suppression of resonance")


def demo_epistemic_gate():
    """Demonstrate epistemic gating with backtracking."""
    print_header("Demo 3: Epistemic Gate - Quality Threshold Enforcement")

    # Current safe state
    psi_current = np.array([1.0, 0.0, 0.0])

    # Target (ground truth)
    psi_target = np.array([1.0, 0.0, 0.0])

    # Risky proposed update (too far from target)
    psi_risky = np.array([0.0, 1.0, 0.0])

    print("\nCurrent state:  [1.0, 0.0, 0.0]")
    print("Target truth:   [1.0, 0.0, 0.0]")
    print("Risky proposal: [0.0, 1.0, 0.0] (orthogonal to truth)")

    # Check quality without gating
    gate_open, q_risky = check_quality_threshold(psi_risky, psi_target, q_min=0.8)
    print(f"\nQuality of risky state: Q = {q_risky:.4f}")
    print(f"Gate status (Q_min=0.8): {'OPEN ✓' if gate_open else 'CLOSED ✗'}")

    # Apply gating with backtracking
    print("\n--- Applying epistemic gate with backtracking ---")
    psi_safe, q_final, n_backtrack = epistemic_gate(
        psi_current,
        psi_risky,
        psi_target,
        q_min=0.8,
        max_backtrack=8,
    )

    print(f"Backtracks performed: {n_backtrack}")
    print(f"Accepted state: [{psi_safe[0]:.3f}, {psi_safe[1]:.3f}, {psi_safe[2]:.3f}]")
    print(f"Final quality: Q = {q_final:.4f}")
    print("\nInterpretation: Gate forced backtracking to maintain Q ≥ 0.8")


def demo_integrated_loop():
    """Demonstrate integrated update loop with all components."""
    print_header("Demo 4: Integrated System - Complete Update Cycle")

    # Initialize system
    psi_state = np.array([1.0, 0.0, 0.0, 0.0])
    psi_target = np.array([0.9, 0.3, 0.2, 0.1])
    q_min = 0.6

    print("Simulating 5 update steps toward a target state...")
    print(f"Target: [{psi_target[0]:.2f}, {psi_target[1]:.2f}, "
          f"{psi_target[2]:.2f}, {psi_target[3]:.2f}]")
    print(f"Quality threshold: Q_min = {q_min}\n")

    for step in range(5):
        # Simulate noisy observation moving toward target
        noise = np.random.normal(0, 0.1, size=4)
        z_obs = 0.7 * psi_target + 0.3 * psi_state + noise

        # Apply VARO update
        psi_proposed = varo_update(psi_state, z_obs, beta=0.5, gamma=0.85)

        # Apply epistemic gate
        psi_accepted, q, n_bt = epistemic_gate(
            psi_state, psi_proposed, psi_target, q_min=q_min
        )

        # Update state
        psi_state = psi_accepted

        print(f"Step {step + 1}: Q = {q:.4f}  |  "
              f"Backtracks: {n_bt}  |  "
              f"State: [{psi_state[0]:.3f}, {psi_state[1]:.3f}, ...]")

    print("\n✓ System successfully converged while maintaining epistemic integrity")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  ALETHEION CORE - Educational Demonstration")
    print("  Semi-Symbolic Coherence Monitoring System")
    print("=" * 70)
    print("\nPaper: 'Aletheion: A Semi-Symbolic Architecture for Internal")
    print("        Coherence Monitoring in Neural Language Systems'")
    print("\nThis demo illustrates the three core mechanisms:")
    print("  1. Q metric - Epistemic quality measurement")
    print("  2. VARO operator - Anti-resonance state updates")
    print("  3. Epistemic gate - Quality threshold enforcement")

    np.random.seed(42)  # For reproducibility

    demo_q_metric()
    demo_varo()
    demo_epistemic_gate()
    demo_integrated_loop()

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
    print("\nFor production-grade implementations with GPU acceleration,")
    print("advanced features, and commercial support, contact:")
    print("  licensing@aletheiaengine.dev")
    print("\n")


if __name__ == "__main__":
    main()
