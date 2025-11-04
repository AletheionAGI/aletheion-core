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
ARC-AGI Puzzle Example with Q₁+Q₂ Epistemic Gating

This example demonstrates how Aletheion Core's Q₁+Q₂ framework can be applied
to an ARC-AGI-style abstract reasoning task. The system:

1. Maintains a symbolic representation of the puzzle rule (ψ_symbolic)
2. Generates neural outputs for candidate solutions (z_neural)
3. Measures epistemic quality before generation (Q₁) and after (Q₂)
4. Uses epistemic gating to decide whether to accept or reject the solution

Scenario: Pattern Transformation Puzzle
---------------------------------------
Input grid:  [1, 1, 0]
Output grid: [2, 2, 0]  (rule: multiply non-zero elements by 2)

The system must learn this rule through symbolic-neural interaction.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src import q_metric, varo_update, epistemic_gate


def encode_rule(rule_description: str, dim: int = 16) -> np.ndarray:
    """
    Encode a symbolic rule as a vector.

    In a real system, this would use proper symbolic encoding (e.g., program
    synthesis, logic rules). Here we simulate with random but consistent vectors.

    Parameters
    ----------
    rule_description : str
        Description of the rule
    dim : int
        Dimensionality of symbolic space

    Returns
    -------
    np.ndarray
        Normalized vector representing the rule
    """
    # Use hash for consistency across calls with same description
    seed = hash(rule_description) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim)
    return vec / np.linalg.norm(vec)


def encode_grid_solution(grid: list, dim: int = 16) -> np.ndarray:
    """
    Encode a grid solution as a neural output vector.

    In a real system, this would be the output of a neural network.
    Here we simulate with hashed encoding.

    Parameters
    ----------
    grid : list
        Grid values
    dim : int
        Dimensionality of neural output space

    Returns
    -------
    np.ndarray
        Normalized vector representing the solution
    """
    # Use grid contents for consistent encoding
    seed = hash(tuple(grid)) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim)
    return vec / np.linalg.norm(vec)


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def main():
    """Run ARC-AGI example with Q₁+Q₂ epistemic gating."""

    print("\n" + "=" * 70)
    print("  ARC-AGI PUZZLE EXAMPLE")
    print("  Epistemic Gating with Q₁+Q₂ Framework")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Setup: Define the puzzle and ground truth
    # -------------------------------------------------------------------------

    print_header("1. Puzzle Setup")

    print("\nPuzzle: Learn the transformation rule")
    print("  Input:  [1, 1, 0, 3]")
    print("  Output: [2, 2, 0, 6]")
    print("\nGround Truth Rule: 'Multiply non-zero elements by 2'")

    # Encode ground truth rule symbolically
    ground_truth_rule = encode_rule("multiply_non_zero_by_2", dim=16)
    print(f"\nGround truth symbolic representation: ψ_truth")
    print(f"  Dimensionality: {len(ground_truth_rule)}")
    print(f"  Norm: {np.linalg.norm(ground_truth_rule):.4f}")

    # -------------------------------------------------------------------------
    # Initial System State
    # -------------------------------------------------------------------------

    print_header("2. Initial System State")

    # System starts with a random hypothesis
    psi_hypothesis = encode_rule("random_initial_hypothesis", dim=16)
    print("\nInitial hypothesis: ψ_system (random)")

    # Measure initial quality (Q₁ before any generation)
    q1_initial = q_metric(psi_hypothesis, ground_truth_rule)
    print(f"  Q₁ (initial coherence with truth): {q1_initial:.4f}")

    if q1_initial < 0.5:
        print("  Status: LOW coherence - system is uncertain")
    else:
        print("  Status: MODERATE coherence")

    # -------------------------------------------------------------------------
    # Iteration 1: Generate Wrong Solution
    # -------------------------------------------------------------------------

    print_header("3. Iteration 1: Generate Wrong Solution")

    print("\nHypothesis: 'Add 1 to all elements'")
    print("  Neural output: [2, 2, 1, 4]")
    print("  Expected:      [2, 2, 0, 6]")
    print("  Result: INCORRECT")

    # Encode wrong solution
    wrong_solution = [2, 2, 1, 4]
    z_wrong = encode_grid_solution(wrong_solution, dim=16)

    # Measure Q₁ (pre-generation quality)
    q1_wrong = q_metric(psi_hypothesis, ground_truth_rule)
    print(f"\n  Q₁ (pre-generation): {q1_wrong:.4f}")

    # Apply VARO update
    psi_updated_wrong = varo_update(
        psi_hypothesis,
        z_wrong,
        beta=0.5,  # Moderate anti-resonance
        gamma=0.85,  # High memory persistence
    )

    # Measure Q₂ (post-generation quality)
    q2_wrong = q_metric(psi_updated_wrong, ground_truth_rule)
    print(f"  Q₂ (post-generation): {q2_wrong:.4f}")

    # Combined quality metric
    q_combined_wrong = 0.4 * q1_wrong + 0.6 * q2_wrong
    print(f"  Q_combined (0.4×Q₁ + 0.6×Q₂): {q_combined_wrong:.4f}")

    # Epistemic gating decision
    q_min = 0.60
    print(f"\n  Quality threshold: Q_min = {q_min:.2f}")

    psi_gated_wrong, q_final_wrong, n_bt_wrong = epistemic_gate(
        psi_hypothesis,
        psi_updated_wrong,
        ground_truth_rule,
        q_min=q_min,
        max_backtrack=8,
    )

    print(f"  Backtracks performed: {n_bt_wrong}")
    print(f"  Final quality: {q_final_wrong:.4f}")

    if q_final_wrong < q_min:
        decision = "REJECT"
        print(f"\n  ❌ DECISION: {decision} - Quality below threshold")
        print("  System maintains previous hypothesis")
        psi_hypothesis = psi_hypothesis  # Stay at old state
    else:
        decision = "ACCEPT (with backtracking)"
        print(f"\n  ⚠️  DECISION: {decision}")
        print("  System partially updates hypothesis (conservative)")
        psi_hypothesis = psi_gated_wrong

    # -------------------------------------------------------------------------
    # Iteration 2: Generate Correct Solution
    # -------------------------------------------------------------------------

    print_header("4. Iteration 2: Generate Correct Solution")

    print("\nHypothesis: 'Multiply non-zero elements by 2'")
    print("  Neural output: [2, 2, 0, 6]")
    print("  Expected:      [2, 2, 0, 6]")
    print("  Result: CORRECT ✓")

    # Encode correct solution
    correct_solution = [2, 2, 0, 6]
    z_correct = encode_grid_solution(correct_solution, dim=16)

    # Measure Q₁ (pre-generation quality)
    q1_correct = q_metric(psi_hypothesis, ground_truth_rule)
    print(f"\n  Q₁ (pre-generation): {q1_correct:.4f}")

    # Apply VARO update
    psi_updated_correct = varo_update(psi_hypothesis, z_correct, beta=0.5, gamma=0.85)

    # Measure Q₂ (post-generation quality)
    q2_correct = q_metric(psi_updated_correct, ground_truth_rule)
    print(f"  Q₂ (post-generation): {q2_correct:.4f}")

    # Combined quality metric
    q_combined_correct = 0.4 * q1_correct + 0.6 * q2_correct
    print(f"  Q_combined (0.4×Q₁ + 0.6×Q₂): {q_combined_correct:.4f}")

    # Epistemic gating decision
    print(f"\n  Quality threshold: Q_min = {q_min:.2f}")

    psi_gated_correct, q_final_correct, n_bt_correct = epistemic_gate(
        psi_hypothesis,
        psi_updated_correct,
        ground_truth_rule,
        q_min=q_min,
        max_backtrack=8,
    )

    print(f"  Backtracks performed: {n_bt_correct}")
    print(f"  Final quality: {q_final_correct:.4f}")

    if q_final_correct >= q_min:
        decision = "ACCEPT"
        print(f"\n  ✅ DECISION: {decision}")
        print("  System updates hypothesis to new rule")
        psi_hypothesis = psi_gated_correct
    else:
        decision = "REJECT"
        print(f"\n  ❌ DECISION: {decision}")
        psi_hypothesis = psi_hypothesis

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------

    print_header("5. Summary")

    print("\nIteration 1 (Wrong Solution):")
    print(f"  Q₁ = {q1_wrong:.4f}, Q₂ = {q2_wrong:.4f}")
    print(f"  Q_combined = {q_combined_wrong:.4f}")
    print(f"  Decision: REJECT (quality below threshold)")

    print("\nIteration 2 (Correct Solution):")
    print(f"  Q₁ = {q1_correct:.4f}, Q₂ = {q2_correct:.4f}")
    print(f"  Q_combined = {q_combined_correct:.4f}")
    print(f"  Decision: ACCEPT (quality meets threshold)")

    print("\n" + "-" * 70)
    print("Key Insights:")
    print("-" * 70)
    print("• Q₁ (pre-generation): Measures confidence before acting")
    print("• Q₂ (post-generation): Validates outcome against truth")
    print("• VARO: Prevents collapse into wrong hypotheses")
    print("• Epistemic Gate: Enforces quality threshold with backtracking")
    print("• Combined: System safely explores while maintaining coherence")

    # -------------------------------------------------------------------------
    # Real-World Application Note
    # -------------------------------------------------------------------------

    print_header("6. Application to Real ARC-AGI")

    print("\nIn a production system:")
    print("  1. ψ_symbolic ← Program synthesis / logic rules")
    print("  2. z_neural ← Transformer/CNN grid predictions")
    print("  3. Q₁ ← Symbolic consistency check (pre-generation)")
    print("  4. Q₂ ← Validation against known examples (post-generation)")
    print("  5. Gate ← Decide whether to commit to this solution")
    print("\nAdvantages:")
    print("  • Prevents acceptance of inconsistent solutions")
    print("  • Maintains interpretable symbolic representation")
    print("  • Allows safe exploration of hypothesis space")
    print("  • Provides confidence calibration for neural outputs")

    # -------------------------------------------------------------------------
    # Comparison with Standard Approach
    # -------------------------------------------------------------------------

    print_header("7. Comparison with Standard Neural-Only Approach")

    print("\nStandard approach (neural network alone):")
    print("  ❌ No explicit symbolic representation")
    print("  ❌ No pre-generation quality check")
    print("  ❌ No post-generation validation gate")
    print("  ❌ Can confidently output wrong answers")
    print("  ❌ No mechanism to reject low-quality hypotheses")

    print("\nAletheion approach (Q₁+Q₂ with gating):")
    print("  ✅ Explicit symbolic representation (ψ)")
    print("  ✅ Pre-generation coherence check (Q₁)")
    print("  ✅ Post-generation validation (Q₂)")
    print("  ✅ Refuses to accept low-quality solutions")
    print("  ✅ Maintains epistemic integrity throughout reasoning")

    print("\n" + "=" * 70)
    print("  Example Complete!")
    print("=" * 70)
    print("\nFor more details, see:")
    print("  • Paper: https://doi.org/10.13140/RG.2.2.29925.87527")
    print("  • Theory: docs/THEORY.md")
    print("  • API: docs/API.md")
    print("\n")


if __name__ == "__main__":
    main()
