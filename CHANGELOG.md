# Changelog

All notable changes to the Aletheion Core project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-15

### Added

#### Core Implementation
- **Q Metric**: Epistemic quality measure mapping cosine similarity to [0,1] range
  - `q_metric()`: Compute coherence between symbolic states
  - `cosine_similarity()`: Compute similarity between vectors
  - `normalize_vector()`: Normalize vectors to unit length

- **VARO Operator**: Variational Anti-Resonance Operator for state updates
  - `varo_update()`: Apply anti-resonance to prevent cognitive collapse
  - `lambda_mu_to_beta_gamma()`: Convert theoretical to implementation parameters
  - Support for configurable anti-resonance strength (β) and memory persistence (γ)

- **Epistemic Gating**: Quality threshold enforcement with backtracking
  - `check_quality_threshold()`: Verify state meets minimum quality threshold
  - `epistemic_gate()`: Apply gating with automatic backtracking
  - Configurable quality thresholds (Q_min) and backtrack parameters

#### Documentation
- Comprehensive README with usage examples and theory overview
- API documentation for all public functions
- Mathematical theory documentation with formulas
- Contributing guidelines for developers
- Citation file (CITATION.cff) for academic use

#### Examples
- `simple_demo.py`: Educational demonstration of all three core components
- `arc_example.py`: Practical example using Q₁+Q₂ for puzzle-solving scenarios

#### Testing
- Unit tests for Q metric (range, identical, orthogonal, opposite vectors)
- Unit tests for VARO operator (normalization, anti-resonance, temporal inertia)
- Unit tests for epistemic gating (accept/reject, thresholding, backtracking)
- Integration tests for combined system behavior
- CI/CD pipeline with GitHub Actions (Python 3.8-3.12)

#### Packaging
- PyPI-ready package configuration (setup.py, pyproject.toml)
- Modern PEP 517/518 build system
- Development dependencies (pytest, black, isort, mypy)
- Documentation dependencies (sphinx)
- Comprehensive MANIFEST.in for distribution

### Technical Details
- **License**: GNU Affero General Public License v3.0 (AGPL-3.0)
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Dependencies**: numpy>=1.20.0, typing-extensions>=4.0.0
- **Paper DOI**: [10.13140/RG.2.2.29925.87527](https://doi.org/10.13140/RG.2.2.29925.87527)

### Notes
This is the initial educational release of Aletheion Core, providing a simplified
reference implementation of the core symbolic coherence monitoring system described
in the paper "Aletheion: A Semi-Symbolic Architecture for Internal Coherence
Monitoring in Neural Language Systems."

For production-grade implementations with GPU acceleration, advanced features,
and commercial support, contact: licensing@aletheiaengine.dev

---

## [Unreleased]

### Planned Features
- GPU acceleration for large-scale applications
- Extended symbolic operators beyond VARO
- Integration examples with popular LLM frameworks
- Performance benchmarks and optimization guides
- Advanced visualization tools for Q metric evolution
- Multi-modal symbolic representation support

---

**Legend:**
- `Added`: New features
- `Changed`: Changes in existing functionality
- `Deprecated`: Soon-to-be removed features
- `Removed`: Removed features
- `Fixed`: Bug fixes
- `Security`: Vulnerability patches
