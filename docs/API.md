# API Reference

Complete API documentation for Aletheion Core. All functions are accessible via the main module:

```python
from src import (
    q_metric,
    cosine_similarity,
    normalize_vector,
    varo_update,
    lambda_mu_to_beta_gamma,
    epistemic_gate,
    check_quality_threshold,
)
```

Or import from submodules:

```python
from src.q_metric import q_metric, cosine_similarity, normalize_vector
from src.varo import varo_update, lambda_mu_to_beta_gamma
from src.epistemic_gate import epistemic_gate, check_quality_threshold
```

---

## Module: `q_metric`

Epistemic quality measurement based on cosine similarity.

### `q_metric(psi_s, psi_t) → float`

Compute the Q metric between two symbolic states.

**Parameters**:
- `psi_s` (np.ndarray): Source symbolic state (system representation)
- `psi_t` (np.ndarray): Target symbolic state (ground truth or goal)

**Returns**:
- `float`: Quality metric Q in [0, 1]
  - Q = 1.0: Perfect alignment
  - Q = 0.5: Orthogonal (neutral)
  - Q = 0.0: Complete opposition

**Examples**:
```python
import numpy as np
from src import q_metric

# Perfect alignment
psi_s = np.array([1.0, 0.0, 0.0])
psi_t = np.array([1.0, 0.0, 0.0])
q = q_metric(psi_s, psi_t)
print(f"Q = {q}")  # Q = 1.0

# Orthogonal vectors
psi_s = np.array([1.0, 0.0, 0.0])
psi_t = np.array([0.0, 1.0, 0.0])
q = q_metric(psi_s, psi_t)
print(f"Q = {q}")  # Q = 0.5

# Opposite vectors
psi_s = np.array([1.0, 0.0, 0.0])
psi_t = np.array([-1.0, 0.0, 0.0])
q = q_metric(psi_s, psi_t)
print(f"Q = {q}")  # Q = 0.0
```

**Formula**:
```
Q(ψ_s, ψ_t) = (1 + cos(ψ_s, ψ_t)) / 2
```

**Notes**:
- Vectors are automatically normalized to unit length
- Works with any dimensionality (must match between inputs)
- Handles zero vectors gracefully (returns Q = 0.5)

---

### `cosine_similarity(a, b) → float`

Compute cosine similarity between two vectors.

**Parameters**:
- `a` (np.ndarray): First vector
- `b` (np.ndarray): Second vector

**Returns**:
- `float`: Cosine similarity in [-1, 1]

**Examples**:
```python
import numpy as np
from src import cosine_similarity

a = np.array([1.0, 1.0])
b = np.array([1.0, 0.0])
cos_sim = cosine_similarity(a, b)
print(f"Cosine similarity = {cos_sim:.3f}")  # ≈ 0.707 (cos 45°)
```

**Formula**:
```
cos(a, b) = (a · b) / (||a|| × ||b||)
```

---

### `normalize_vector(vector, eps=1e-12) → np.ndarray`

Normalize a vector to unit length.

**Parameters**:
- `vector` (np.ndarray): Input vector to normalize
- `eps` (float, optional): Small constant to prevent division by zero (default: 1e-12)

**Returns**:
- `np.ndarray`: Unit-normalized vector

**Examples**:
```python
import numpy as np
from src import normalize_vector

v = np.array([3.0, 4.0])
v_norm = normalize_vector(v)
print(v_norm)  # [0.6, 0.8]
print(np.linalg.norm(v_norm))  # 1.0
```

**Notes**:
- If ||vector|| < eps, returns zero vector
- Preserves direction, changes only magnitude

---

## Module: `varo`

Variational Anti-Resonance Operator for state updates.

### `varo_update(psi_prev, z_new, beta=0.5, gamma=0.9, eta=0.0) → np.ndarray`

Update symbolic state using the Variational Anti-Resonance Operator.

**Parameters**:
- `psi_prev` (np.ndarray): Previous symbolic state (ψ at time t-1)
- `z_new` (np.ndarray): New symbolic observation (z at time t)
- `beta` (float, optional): Anti-resonance strength in [0, 1] (default: 0.5)
  - β = 0: No suppression (standard EMA)
  - β = 1: Full suppression of parallel component
- `gamma` (float, optional): Memory persistence factor in [0, 1] (default: 0.9)
  - γ = 0: No memory (replace state completely)
  - γ = 1: Full memory (no update)
- `eta` (float, optional): Noise injection strength for exploration (default: 0.0)

**Returns**:
- `np.ndarray`: Updated symbolic state ψ_new (normalized to unit length)

**Examples**:
```python
import numpy as np
from src import varo_update

# Standard EMA (no anti-resonance)
psi = np.array([1.0, 0.0, 0.0])
z = np.array([0.9, 0.1, 0.0])
psi_new = varo_update(psi, z, beta=0.0, gamma=0.9)
print(psi_new)  # Close to [1.0, 0.0, 0.0], slightly toward z

# With anti-resonance
psi = np.array([1.0, 0.0, 0.0])
z = np.array([1.0, 0.0, 0.0])  # Same direction
psi_new = varo_update(psi, z, beta=0.5, gamma=0.9)
# Suppresses parallel component, maintains exploration

# Full anti-resonance
psi = np.array([1.0, 0.0, 0.0])
z = np.array([1.0, 0.0, 0.0])
psi_new = varo_update(psi, z, beta=1.0, gamma=0.9)
# Maximum suppression of resonance
```

**Algorithm**:
```
1. anti = z - β × (z · ψ̂) × ψ̂     (suppress parallel component)
2. ψ_new = (1 - γ) × anti + γ × ψ̂  (blend with memory)
3. ψ_new = ψ_new / ||ψ_new||        (normalize)
```

**Notes**:
- Prevents cognitive collapse by suppressing resonant components
- Maintains memory continuity through exponential moving average
- Always returns normalized (unit-length) vectors
- Setting β=0 reduces to standard EMA

---

### `lambda_mu_to_beta_gamma(lambda_ar, mu_ar) → (float, float)`

Convert variational parameters (λ, μ) to implementation parameters (β, γ).

**Parameters**:
- `lambda_ar` (float): Anti-resonance intensity parameter (λ ≥ 0)
- `mu_ar` (float): Memory persistence parameter (μ ≥ 0)

**Returns**:
- `beta` (float): Anti-resonance strength in [0, 1)
- `gamma` (float): Memory persistence factor in [0, 1)

**Examples**:
```python
from src import lambda_mu_to_beta_gamma

# Convert theoretical parameters to implementation parameters
beta, gamma = lambda_mu_to_beta_gamma(lambda_ar=1.0, mu_ar=9.0)
print(f"β = {beta:.3f}, γ = {gamma:.3f}")  # β=0.909, γ=0.900
```

**Mapping**:
```
γ = 1 - 1/(1 + μ)
β = 1 - 1/(1 + λ + μ)
```

**Notes**:
- Provides connection between variational formulation and implementation
- Both λ and μ must be non-negative
- Results are clamped to [0, 0.999999] for numerical stability

---

## Module: `epistemic_gate`

Quality threshold enforcement with backtracking.

### `check_quality_threshold(psi_state, psi_target, q_min=0.5) → (bool, float)`

Check if the symbolic state meets the minimum quality threshold.

**Parameters**:
- `psi_state` (np.ndarray): Current or proposed symbolic state
- `psi_target` (np.ndarray): Target symbolic state (ground truth or goal)
- `q_min` (float, optional): Minimum acceptable quality threshold in [0, 1] (default: 0.5)

**Returns**:
- `gate_open` (bool): True if quality ≥ q_min (accept state), False otherwise
- `quality` (float): The computed Q metric value

**Examples**:
```python
import numpy as np
from src import check_quality_threshold

# High quality state (gate open)
psi_state = np.array([1.0, 0.0, 0.0])
psi_target = np.array([0.9, 0.1, 0.0])
gate_open, q = check_quality_threshold(psi_state, psi_target, q_min=0.5)
print(f"Gate open: {gate_open}, Q={q:.3f}")  # Gate open: True, Q≈0.995

# Low quality state (gate closed)
psi_state = np.array([1.0, 0.0, 0.0])
psi_target = np.array([-1.0, 0.0, 0.0])
gate_open, q = check_quality_threshold(psi_state, psi_target, q_min=0.5)
print(f"Gate open: {gate_open}, Q={q:.3f}")  # Gate open: False, Q≈0.0
```

**Notes**:
- Simple binary decision: accept or reject
- Does not modify state, only checks quality
- Use `epistemic_gate()` for automatic backtracking

---

### `epistemic_gate(psi_current, psi_proposed, psi_target, q_min=0.5, max_backtrack=8, backtrack_factor=0.5) → (np.ndarray, float, int)`

Apply epistemic gating with automatic backtracking.

**Parameters**:
- `psi_current` (np.ndarray): Current symbolic state (safe baseline)
- `psi_proposed` (np.ndarray): Proposed new symbolic state (to be validated)
- `psi_target` (np.ndarray): Target symbolic state (ground truth or goal)
- `q_min` (float, optional): Minimum acceptable quality threshold (default: 0.5)
- `max_backtrack` (int, optional): Maximum number of backtracking iterations (default: 8)
- `backtrack_factor` (float, optional): Factor to reduce step size on each backtrack (default: 0.5)

**Returns**:
- `psi_accepted` (np.ndarray): Accepted state (either proposed or backtracked version)
- `final_quality` (float): Quality of the accepted state
- `num_backtracks` (int): Number of backtracking iterations performed

**Examples**:
```python
import numpy as np
from src import epistemic_gate

# Accepted proposal (no backtracking)
current = np.array([1.0, 0.0, 0.0])
proposed = np.array([0.95, 0.05, 0.0])
target = np.array([1.0, 0.0, 0.0])
accepted, q, n = epistemic_gate(current, proposed, target, q_min=0.8)
print(f"Backtracks: {n}, Q={q:.3f}")  # Backtracks: 0, Q≈0.999

# Rejected proposal (requires backtracking)
current = np.array([1.0, 0.0, 0.0])
proposed = np.array([0.0, 1.0, 0.0])  # Orthogonal
target = np.array([1.0, 0.0, 0.0])
accepted, q, n = epistemic_gate(current, proposed, target, q_min=0.8)
print(f"Backtracks: {n}, Q={q:.3f}")  # Backtracks: 3, Q≈0.854
print(accepted)  # Interpolated between current and proposed
```

**Algorithm**:
```
1. Check if Q(proposed, target) ≥ Q_min
   → If yes: Accept proposed, return (proposed, Q, 0)

2. If no, start backtracking:
   step_size = 1.0
   for i in range(max_backtrack):
       step_size *= backtrack_factor
       candidate = (1 - step_size) × current + step_size × proposed
       candidate = candidate / ||candidate||

       if Q(candidate, target) ≥ Q_min:
           return (candidate, Q, i+1)

3. If all backtracks fail:
   return (current, Q(current, target), max_backtrack)
```

**Notes**:
- Automatically finds safe intermediate state between current and proposed
- Guarantees returned state meets quality threshold (or returns current state)
- Number of backtracks indicates how risky the proposed update was
- Always returns normalized vectors

---

## Complete Usage Example

Here's a complete example integrating all components:

```python
import numpy as np
from src import q_metric, varo_update, epistemic_gate

# Initialize system
np.random.seed(42)
psi_state = np.array([1.0, 0.0, 0.0, 0.0])
psi_target = np.array([0.9, 0.3, 0.2, 0.1])
psi_target = psi_target / np.linalg.norm(psi_target)

# Parameters
beta = 0.5        # Anti-resonance strength
gamma = 0.85      # Memory persistence
q_min = 0.6       # Quality threshold

print("Initial state:", psi_state)
print("Target state:", psi_target)
print()

# Run update loop
for step in range(5):
    # 1. Measure pre-generation quality (Q₁)
    q1 = q_metric(psi_state, psi_target)

    # 2. Simulate new observation (moving toward target + noise)
    z_obs = 0.7 * psi_target + 0.3 * psi_state + np.random.normal(0, 0.1, size=4)

    # 3. Apply VARO update
    psi_proposed = varo_update(psi_state, z_obs, beta=beta, gamma=gamma)

    # 4. Measure post-generation quality (Q₂)
    q2 = q_metric(psi_proposed, psi_target)

    # 5. Apply epistemic gate
    psi_accepted, q_final, n_bt = epistemic_gate(
        psi_state, psi_proposed, psi_target, q_min=q_min
    )

    # 6. Update state
    psi_state = psi_accepted

    print(f"Step {step + 1}:")
    print(f"  Q₁ = {q1:.4f}  (pre-generation)")
    print(f"  Q₂ = {q2:.4f}  (post-generation)")
    print(f"  Q_final = {q_final:.4f}")
    print(f"  Backtracks: {n_bt}")
    print(f"  State: [{psi_state[0]:.3f}, {psi_state[1]:.3f}, ...]")
    print()

print("✓ Update loop complete")
print(f"Final quality: Q = {q_metric(psi_state, psi_target):.4f}")
```

---

## Type Signatures

For type checking with mypy:

```python
from typing import Tuple
import numpy as np

# q_metric module
def q_metric(psi_s: np.ndarray, psi_t: np.ndarray) -> float: ...
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float: ...
def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray: ...

# varo module
def varo_update(
    psi_prev: np.ndarray,
    z_new: np.ndarray,
    beta: float = 0.5,
    gamma: float = 0.9,
    eta: float = 0.0,
) -> np.ndarray: ...

def lambda_mu_to_beta_gamma(
    lambda_ar: float,
    mu_ar: float,
) -> Tuple[float, float]: ...

# epistemic_gate module
def check_quality_threshold(
    psi_state: np.ndarray,
    psi_target: np.ndarray,
    q_min: float = 0.5,
) -> Tuple[bool, float]: ...

def epistemic_gate(
    psi_current: np.ndarray,
    psi_proposed: np.ndarray,
    psi_target: np.ndarray,
    q_min: float = 0.5,
    max_backtrack: int = 8,
    backtrack_factor: float = 0.5,
) -> Tuple[np.ndarray, float, int]: ...
```

---

## Performance Considerations

### Computational Complexity

- **q_metric**: O(d) where d = vector dimensionality
- **varo_update**: O(d) per update
- **epistemic_gate**: O(d × n_backtrack) worst case, typically O(d)

### Memory Usage

All functions operate on numpy arrays with minimal memory overhead:
- Single state vector: O(d) floats
- No internal caching or accumulation

### Optimization Tips

1. **Use appropriate dtype**: `np.float32` is sufficient for most applications and faster than `np.float64`

2. **Vectorize batch updates**: Process multiple states in parallel
   ```python
   # Batch processing (N states)
   psi_batch = np.array([...])  # Shape: (N, d)
   z_batch = np.array([...])    # Shape: (N, d)

   # Vectorized VARO (requires modification)
   # See examples/ for batch implementations
   ```

3. **Precompute target normalization**: If target is fixed, normalize once
   ```python
   psi_target = normalize_vector(psi_target)  # Do once
   # Reuse in loop
   ```

4. **Use in-place operations**: For very large dimensions, consider in-place variants

---

## Error Handling

All functions handle edge cases gracefully:

- **Empty arrays**: Return appropriate empty/zero results
- **Zero vectors**: Normalize to zero vector (avoids division by zero)
- **Dimension mismatches**: Automatically flatten and process
- **Out-of-range parameters**: Clamp to valid ranges (where applicable)

No exceptions are raised for normal usage; functions return sensible defaults for degenerate inputs.

---

## See Also

- [THEORY.md](THEORY.md) - Mathematical foundations and formulas
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
- [../examples/](../examples/) - Complete working examples
- [../tests/](../tests/) - Unit tests demonstrating usage

---

**Paper**: [Aletheion: A Semi-Symbolic Architecture](https://doi.org/10.13140/RG.2.2.29925.87527)
**Repository**: https://github.com/AletheionAGI/aletheion-core
**License**: AGPL-3.0
