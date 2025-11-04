# Mathematical Theory and Formulation

This document provides a detailed mathematical explanation of the core components in Aletheion Core. For the complete theoretical foundation, please refer to the full paper:

**Paper**: "Aletheion: A Semi-Symbolic Architecture for Internal Coherence Monitoring in Neural Language Systems"
**DOI**: [10.13140/RG.2.2.29925.87527](https://doi.org/10.13140/RG.2.2.29925.87527)
**PDF**: [Aletheion_Preprint___v4_0.pdf](../Aletheion_Preprint___v4_0.pdf)

---

## Overview

The Aletheion architecture combines three fundamental mechanisms for maintaining epistemic coherence in symbolic AI systems:

1. **Q Metric** - Measures epistemic quality
2. **VARO Operator** - Updates state while preventing collapse
3. **Epistemic Gating** - Enforces quality thresholds

---

## 1. Q Metric: Epistemic Quality Measure

### Definition

The Q metric quantifies the epistemic coherence between two symbolic state vectors by mapping cosine similarity to the interval [0, 1]:

```
Q(ψ_s, ψ_t) = (1 + cos(ψ_s, ψ_t)) / 2
```

Where:
- `ψ_s` is the system's symbolic representation (source state)
- `ψ_t` is the target symbolic state (ground truth or goal)
- `cos(ψ_s, ψ_t)` is the cosine similarity between the vectors

### Mathematical Properties

**Range**: Q ∈ [0, 1]

**Interpretation**:
- `Q = 1.0`: Perfect alignment (vectors point in same direction)
- `Q = 0.5`: Orthogonal vectors (no correlation)
- `Q = 0.0`: Complete opposition (vectors point in opposite directions)

**Cosine Similarity**:
```
cos(ψ_s, ψ_t) = (ψ_s · ψ_t) / (||ψ_s|| × ||ψ_t||)
```

Where:
- `·` denotes the dot product
- `||·||` denotes the Euclidean norm (L2 norm)

### Key Characteristics

1. **Symmetry**: Q(ψ_s, ψ_t) = Q(ψ_t, ψ_s)
2. **Normalization Invariance**: Q depends only on the direction of vectors, not their magnitude
3. **Continuity**: Small changes in vectors produce small changes in Q
4. **Monotonicity**: Q increases monotonically with the angle between vectors

### Implementation Note

The Q metric is computed on normalized (unit-length) vectors to ensure consistency:

```python
ψ_s_normalized = ψ_s / ||ψ_s||
ψ_t_normalized = ψ_t / ||ψ_t||
Q = 0.5 * (1.0 + dot(ψ_s_normalized, ψ_t_normalized))
```

---

## 2. VARO Operator: Variational Anti-Resonance Operator

### Problem Statement

Standard exponential moving average (EMA) updates can lead to **cognitive collapse**, where the system's representation becomes trapped in a dominant mode, losing the ability to explore alternative hypotheses.

### Solution: Anti-Resonance

The VARO operator suppresses the component of new observations that aligns (resonates) with the current state, maintaining cognitive flexibility:

```
ψ' = VARO(ψ, z, β, γ)
```

### Full Update Formula

The VARO update consists of three steps:

**Step 1: Anti-Resonance Projection**
```
z_anti = z - β × (z · ψ̂) × ψ̂
```

Where:
- `z` is the new observation (neural output or symbolic update)
- `ψ̂` is the normalized previous state: `ψ̂ = ψ / ||ψ||`
- `β ∈ [0, 1]` is the anti-resonance strength
- `(z · ψ̂)` projects z onto the current state direction
- `β × (z · ψ̂) × ψ̂` is the resonant component to suppress

**Step 2: Temporal Blending (EMA)**
```
ψ_new = (1 - γ) × z_anti + γ × ψ̂
```

Where:
- `γ ∈ [0, 1]` is the memory persistence factor
- Higher γ → more memory retention
- Lower γ → faster adaptation to new observations

**Step 3: Normalization**
```
ψ' = ψ_new / ||ψ_new||
```

Ensures the symbolic state remains on the unit hypersphere.

### Parameter Interpretation

**Anti-Resonance Strength (β)**:
- `β = 0`: Standard EMA (no anti-resonance, risk of collapse)
- `β = 0.5`: Moderate suppression (balanced exploration)
- `β = 1`: Full suppression of parallel component (maximum anti-resonance)

**Memory Persistence (γ)**:
- `γ = 0`: No memory (immediate replacement: ψ' = z_anti)
- `γ = 0.9`: High memory (slow adaptation, typical value)
- `γ = 1`: Perfect memory (no update, ψ' = ψ̂)

### Variational Form

The VARO operator can be derived from a variational principle with Lagrangian:

```
L(λ, μ) = ||z - ψ'||² + λ||(z · ψ̂)ψ̂||² + μ||ψ' - ψ̂||²
```

Where λ and μ are variational parameters. The mapping to implementation parameters is:

```
γ = 1 - 1/(1 + μ)
β = 1 - 1/(1 + λ + μ)
```

This establishes a connection between the phenomenological update rule and a principled variational objective.

### Geometric Interpretation

VARO projects the observation `z` onto the subspace orthogonal to the current state `ψ`, then blends this orthogonal component with the current state. This ensures continuous exploration of the symbolic space while maintaining temporal continuity.

---

## 3. Epistemic Gating: Quality Threshold Enforcement

### Motivation

Even with VARO, proposed updates may sometimes produce states with unacceptably low epistemic quality (low Q). The epistemic gate enforces a minimum quality threshold.

### Basic Gating Rule

```
Accept(ψ_proposed) = {
    True   if Q(ψ_proposed, ψ_target) ≥ Q_min
    False  otherwise
}
```

Where:
- `Q_min ∈ [0, 1]` is the minimum acceptable quality threshold
- Typical values: Q_min ∈ [0.6, 0.85]

### Backtracking Mechanism

When a proposed state fails the gate, the system performs **backtracking** by interpolating between the safe current state and the risky proposed state:

**Backtracking Formula**:
```
ψ_safe = (1 - α) × ψ_current + α × ψ_proposed
ψ_safe = ψ_safe / ||ψ_safe||
```

Where:
- `α ∈ [0, 1]` is the step size
- `α = 1`: Full step (accept proposed state)
- `α = 0`: No step (remain at current state)

**Iterative Backtracking**:
```
α_0 = 1.0
α_{i+1} = α_i × δ

For i = 1, 2, ..., max_backtrack:
    ψ_candidate = (1 - α_i) × ψ_current + α_i × ψ_proposed
    ψ_candidate = ψ_candidate / ||ψ_candidate||

    if Q(ψ_candidate, ψ_target) ≥ Q_min:
        Accept ψ_candidate
        Break
```

Where:
- `δ ∈ (0, 1)` is the backtrack factor (typically 0.5)
- `max_backtrack` is the maximum number of iterations (typically 8)

### Fallback Strategy

If backtracking fails to find an acceptable state after `max_backtrack` iterations, the gate returns the current state (safest option):

```
ψ_accepted = ψ_current
```

This ensures the system never accepts states below the quality threshold.

---

## 4. Integrated Update Loop

The complete update cycle combines all three components:

### Single Update Step

Given:
- Current state: `ψ_t`
- New observation: `z_t`
- Target state: `ψ_target`
- Parameters: `β, γ, Q_min`

**Algorithm**:
```
1. Compute Q₁ (pre-generation quality):
   Q₁ = Q(ψ_t, ψ_target)

2. Apply VARO update:
   ψ_proposed = VARO(ψ_t, z_t, β, γ)

3. Compute Q₂ (post-generation quality):
   Q₂ = Q(ψ_proposed, ψ_target)

4. Apply epistemic gate:
   (ψ_{t+1}, Q_final, n_backtrack) = Gate(ψ_t, ψ_proposed, ψ_target, Q_min)

5. Update state:
   ψ_t ← ψ_{t+1}
```

### Q₁ + Q₂ Framework

The dual quality measurement (Q₁ before update, Q₂ after update) provides two critical signals:

- **Q₁**: Measures current alignment with target (pre-generation confidence)
- **Q₂**: Measures proposed alignment with target (post-generation validation)

Combined metric:
```
Q_combined = w₁ × Q₁ + w₂ × Q₂
```

Where typical weights are `w₁ = 0.4, w₂ = 0.6` (more weight on post-generation quality).

---

## 5. Convergence and Stability

### Convergence Properties

Under mild conditions (bounded observations, bounded anti-resonance), the VARO operator converges to a stable distribution on the unit hypersphere.

**Lyapunov Function**:
```
V(ψ) = -log Q(ψ, ψ_target) = -log((1 + cos(ψ, ψ_target))/2)
```

For most parameter settings, V(ψ) decreases on average, ensuring convergence toward the target.

### Stability Analysis

The system maintains stability through:

1. **Normalization**: Keeps states on unit sphere (bounded)
2. **Gating**: Prevents large, destabilizing jumps
3. **Memory (γ)**: Provides temporal smoothing
4. **Anti-resonance (β)**: Prevents mode collapse

---

## 6. Practical Considerations

### Parameter Selection Guidelines

**Anti-Resonance (β)**:
- Exploration-heavy tasks: β ∈ [0.3, 0.7]
- Exploitation-heavy tasks: β ∈ [0.0, 0.3]
- Novel environments: β ∈ [0.5, 0.9]

**Memory Persistence (γ)**:
- Fast-changing environments: γ ∈ [0.7, 0.85]
- Stable environments: γ ∈ [0.85, 0.95]
- Noisy observations: γ ∈ [0.90, 0.95]

**Quality Threshold (Q_min)**:
- Safety-critical applications: Q_min ∈ [0.80, 0.90]
- Standard applications: Q_min ∈ [0.60, 0.75]
- Exploratory applications: Q_min ∈ [0.40, 0.60]

### Computational Complexity

- **Q Metric**: O(d) where d is vector dimensionality
- **VARO Update**: O(d) per update
- **Epistemic Gate**: O(d × n_backtrack) in worst case

All operations scale linearly with dimensionality, making the system efficient for high-dimensional symbolic spaces.

---

## 7. Extensions and Future Work

### Planned Theoretical Extensions

1. **Multi-Target Gating**: Maintain coherence with multiple target states simultaneously
2. **Adaptive Thresholds**: Learn Q_min dynamically from data
3. **Hierarchical Symbolic Spaces**: Apply VARO at multiple levels of abstraction
4. **Continuous-Time Formulation**: Derive differential equation form of VARO
5. **Information-Theoretic Analysis**: Characterize information flow through the gate

### Open Research Questions

- Optimal parameter schedules for different task types
- Theoretical guarantees for convergence rates
- Extension to non-Euclidean symbolic spaces (e.g., hyperbolic embeddings)
- Integration with gradient-based optimization

---

## References

1. **Main Paper**:
   Felipe M. Muniz. "Aletheion: A Semi-Symbolic Architecture for Internal Coherence Monitoring in Neural Language Systems." 2025.
   DOI: [10.13140/RG.2.2.29925.87527](https://doi.org/10.13140/RG.2.2.29925.87527)

2. **Related Work**:
   - Variational inference and gradient flow
   - Anti-Hebbian learning
   - Episodic memory systems
   - Symbolic regression and program synthesis

For implementation details, see [API.md](API.md).
For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
