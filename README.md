# Aletheion Core - Educational Release

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Paper](https://img.shields.io/badge/Paper-Preprint-orange.svg)](https://github.com/AletheionAGI/aletheion-core/Aletheion_Preprint___v4_0.pdf)

**Aletheion Core** is an educational implementation of the semi-symbolic coherence monitoring system described in the paper:

> **"Aletheion: A Semi-Symbolic Architecture for Internal Coherence Monitoring in Neural Language Systems"**
> Felipe M. Muniz
> Preprint v4.0 (2025)

This release contains the three fundamental mechanisms that enable epistemic quality monitoring in neural language systems.

---

## 📚 What's Included

This educational package contains **simplified, pedagogical implementations** of:

### 1. **Q Metric** - Epistemic Quality Measure
Quantifies coherence between symbolic representations in [0, 1]:
```
Q(ψ_s, ψ_t) = (1 + cos(ψ_s, ψ_t)) / 2
```
- Q = 1.0: Perfect alignment (truth)
- Q = 0.5: Orthogonal (neutral)
- Q = 0.0: Complete opposition (falsehood)

**Paper reference:** Equation 1, Section 3.1

### 2. **VARO Operator** - Variational Anti-Resonance Operator
Prevents cognitive collapse by suppressing resonant components:
```
anti = z - β·⟨z, ψ̂⟩·ψ̂
ψ_new = normalize((1-γ)·anti + γ·ψ̂)
```
- β: Anti-resonance strength
- γ: Memory persistence factor

**Paper reference:** Equation 10, Section 3.2

### 3. **Epistemic Gate** - Quality Threshold Enforcement
Ensures updates maintain minimum coherence with target:
- Opens when Q ≥ Q_min (accept update)
- Closes when Q < Q_min (backtrack or reject)
- Automatic interpolation to find acceptable states

**Paper reference:** Section 4.3

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AletheionAGI/aletheion-core.git
cd aletheion-core

# Install dependencies (NumPy only)
pip install numpy
```

### Run the Demo

```bash
python examples/simple_demo.py
```

This demonstrates:
- Q metric calculations for different vector alignments
- VARO updates with varying anti-resonance strengths
- Epistemic gating with automatic backtracking
- Complete integrated update loop

### Run Tests

```bash
# Option 1: Direct execution
python tests/test_basic.py

# Option 2: Using pytest (if installed)
python -m pytest tests/test_basic.py -v
```

---

## 💡 Basic Usage

```python
import numpy as np
from src import q_metric, varo_update, epistemic_gate

# Initialize symbolic states
psi_state = np.array([1.0, 0.0, 0.0])
psi_target = np.array([0.9, 0.3, 0.2])

# Measure epistemic quality
quality = q_metric(psi_state, psi_target)
print(f"Coherence: Q = {quality:.3f}")

# Update state with anti-resonance
z_observation = np.array([0.8, 0.4, 0.1])
psi_new = varo_update(
    psi_prev=psi_state,
    z_new=z_observation,
    beta=0.5,   # Anti-resonance strength
    gamma=0.9   # Memory persistence
)

# Apply epistemic gating
psi_safe, q_final, backtracks = epistemic_gate(
    psi_current=psi_state,
    psi_proposed=psi_new,
    psi_target=psi_target,
    q_min=0.7   # Quality threshold
)

print(f"Accepted state quality: Q = {q_final:.3f}")
print(f"Backtracking steps: {backtracks}")
```

---

## 📖 Paper & Citation

**Full Paper:** [Aletheion_Preprint_v4_0.pdf](https://github.com/AletheionAGI/aletheion-core/Aletheion_Preprint___v4_0.pdf)

If you use this code in your research, please cite:

```bibtex
@article{muniz2025aletheion,
  title={Aletheion: A Semi-Symbolic Architecture for Internal Coherence Monitoring in Neural Language Systems},
  author={Muniz, Felipe M.},
  year={2025},
  note={Preprint v4.0}
}
```

---

## 🎯 Educational Focus

This release prioritizes **clarity and pedagogy** over performance:

✅ **Included:**
- Core mathematical formulations
- Clean, readable Python implementations
- Extensive documentation and examples
- Educational demonstrations
- Basic test suite

❌ **Not Included (Production Features):**
- GPU acceleration (PyTorch/CUDA)
- Neural network integrations
- Production-optimized implementations
- Advanced exploration operators
- Multi-agent coordination
- Proprietary optimizations
- Commercial SaaS features

**Code size:** ~700 lines (excluding tests and demos)

---

## 📁 Project Structure

```
aletheion-core/
├── src/
│   ├── __init__.py          # Package exports
│   ├── q_metric.py          # Q metric & cosine similarity
│   ├── varo.py              # VARO operator & parameter mapping
│   └── epistemic_gate.py    # Quality threshold enforcement
├── examples/
│   └── simple_demo.py       # Interactive demonstration
├── tests/
│   └── test_basic.py        # Unit tests
└── README.md                # This file
```

---

## 🔬 Research Context

The Aletheion architecture addresses a fundamental challenge in AGI: **how can neural systems maintain internal coherence while remaining epistemically honest?**

Traditional approaches either:
- Optimize for consistency → Risk collapse into self-reinforcing loops
- Optimize for exploration → Risk incoherence and hallucination

Aletheion balances these through:
1. **Symbolic quality monitoring** (Q metric)
2. **Anti-resonance updates** (VARO)
3. **Epistemic integrity enforcement** (gating)

This creates a system that can explore conceptual spaces while maintaining coherence with ground truth.

---

## 📜 License

**GNU Affero General Public License v3.0 (AGPL-3.0)**

This educational release is free and open source software. You are free to:
- ✅ Use it for research and education
- ✅ Modify and extend it
- ✅ Share your improvements

**Requirements:**
- Any modifications must also be released under AGPL-3.0
- Network use requires source code disclosure
- Attribution to original author

See [LICENSE](./LICENSE) for full terms.

---

## 💼 Commercial Solutions

For **production deployments**, the full **AletheiaEngine** platform offers:

- 🚀 **GPU-accelerated implementations** (PyTorch/CUDA)
- 🧠 **Neural network integrations** (transformers, LLMs)
- ⚡ **Optimized inference pipelines** (ONNX, TensorRT)
- 🔐 **Enterprise features** (auth, billing, monitoring)
- 🌐 **SaaS deployment** (FastAPI backend, Next.js frontend)
- 📊 **Advanced metrics & telemetry**
- 🛡️ **Production support & SLAs**

**Interested in commercial licensing?**
Contact: **contact@aletheionagi.com**

---

## 🤝 Contributing

This is an educational reference implementation. For contributions:

1. **Bug fixes & documentation:** Welcome via GitHub Issues/PRs
2. **Extensions & research:** Consider publishing separate packages
3. **Commercial features:** Contact us for partnership opportunities

---

## 🙏 Acknowledgments

Developed by Felipe M. Muniz as part of the AletheiaEngine project.

Special thanks to the research community exploring symbolic-neural integration, epistemic coherence, and AGI safety.

---

## 📞 Support & Contact

- **Research questions:** Open a GitHub Issue
- **Paper discussions:** GitHub Discussions
- **Commercial inquiries:** licensing@aletheionagi.com
- **Repository:** https://github.com/AletheionAGI/aletheion-core

---

## 🔗 Related Resources

- [Full AletheiaEngine Repository](https://github.com/AletheionAGI/AletheiaEngine)
- [Paper PDF](https://github.com/AletheionAGI/aletheion-core/blob/main/paper/Aletheion_Preprint___v4_0.pdf)
- [Documentation to be produced]()
---

**Aletheion Core** - Truth as an asymptotic horizon 🜂
