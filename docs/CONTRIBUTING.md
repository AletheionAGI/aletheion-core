# Contributing to Aletheion Core

Thank you for your interest in contributing to Aletheion Core! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Running Tests](#running-tests)
6. [Submitting Changes](#submitting-changes)
7. [Reporting Issues](#reporting-issues)
8. [Feature Requests](#feature-requests)
9. [Documentation](#documentation)
10. [License](#license)

---

## Code of Conduct

This project adheres to a standard code of conduct. By participating, you are expected to:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Prioritize the community's best interests
- Maintain professional discourse

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of symbolic AI and/or the Aletheion paper

### Quick Start

1. **Fork the repository**
   ```bash
   # On GitHub, click "Fork" button
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/aletheion-core.git
   cd aletheion-core
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/AletheionAGI/aletheion-core.git
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "from src import q_metric, varo_update, epistemic_gate; print('✓ Import successful')"
```

### 3. Install Development Tools

The `[dev]` extras include:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatter
- `isort` - Import sorter
- `mypy` - Type checker

### 4. Verify Setup

```bash
# Run tests
pytest tests/

# Check code formatting
black --check src/ tests/ examples/

# Check import sorting
isort --check-only src/ tests/ examples/

# Run type checker
mypy src/
```

---

## Coding Standards

### Code Style

We follow **PEP 8** with the following specifics:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings (consistent with Black)
- **Imports**: Sorted with isort (black profile)

### Formatting with Black

```bash
# Format all Python files
black src/ tests/ examples/

# Check without modifying
black --check src/
```

### Import Sorting with isort

```bash
# Sort imports
isort src/ tests/ examples/

# Check only
isort --check-only src/
```

### Type Hints

- Use type hints for all public functions
- Use `from __future__ import annotations` for forward references
- Example:
  ```python
  from __future__ import annotations
  import numpy as np

  def q_metric(psi_s: np.ndarray, psi_t: np.ndarray) -> float:
      """Compute Q metric."""
      ...
  ```

### Docstrings

Use **Google-style docstrings**:

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Brief description of function.

    Longer description if needed, explaining the purpose
    and behavior of the function.

    Parameters
    ----------
    param1 : type1
        Description of param1
    param2 : type2
        Description of param2

    Returns
    -------
    return_type
        Description of return value

    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected output

    Notes
    -----
    Any additional notes or caveats
    """
    ...
```

### File Headers

All new source files must include the AGPL-3.0 header:

```python
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
```

---

## Running Tests

### Run All Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_q_metric.py

# Run specific test function
pytest tests/test_q_metric.py::test_q_metric_range
```

### Run Tests with Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# View coverage report
# Open htmlcov/index.html in browser
```

### Run Tests for Specific Python Versions

```bash
# Using tox (if configured)
tox

# Or manually with different Python versions
python3.8 -m pytest tests/
python3.9 -m pytest tests/
python3.10 -m pytest tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names: `test_q_metric_returns_one_for_identical_vectors`
- Include docstrings explaining what is being tested

Example test:

```python
def test_q_metric_range():
    """Test that Q metric returns values in [0, 1]."""
    import numpy as np
    from src import q_metric

    # Generate random vectors
    for _ in range(100):
        a = np.random.randn(10)
        b = np.random.randn(10)
        q = q_metric(a, b)

        assert 0.0 <= q <= 1.0, f"Q={q} outside valid range [0, 1]"
```

---

## Submitting Changes

### 1. Keep Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream changes into your main branch
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

### 2. Create a Feature Branch

```bash
# Branch from main
git checkout main
git checkout -b feature/descriptive-name

# Examples:
# feature/add-batch-varo-update
# fix/normalize-zero-vector-bug
# docs/improve-api-examples
```

### 3. Make Your Changes

- Write clear, focused commits
- Follow coding standards
- Add/update tests
- Update documentation if needed

### 4. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add batch processing support for VARO update

- Implement vectorized VARO update for multiple states
- Add tests for batch processing
- Update API documentation with batch examples
"
```

**Commit Message Guidelines**:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description (wrap at 72 chars)
- Use imperative mood: "Add feature" not "Added feature"

### 5. Push to Your Fork

```bash
git push origin feature/descriptive-name
```

### 6. Create Pull Request

1. Go to https://github.com/AletheionAGI/aletheion-core
2. Click "Pull Requests" → "New Pull Request"
3. Click "compare across forks"
4. Select your fork and branch
5. Fill in the PR template:

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- List of specific changes
- With bullet points

## Testing
How have you tested this?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
- [ ] No new warnings

## Related Issues
Fixes #123
```

### 7. Address Review Feedback

- Respond to comments
- Make requested changes
- Push updates to same branch (PR will update automatically)
- Request re-review when ready

---

## Reporting Issues

### Before Submitting an Issue

1. **Search existing issues**: Check if already reported
2. **Try latest version**: Ensure bug exists in current release
3. **Minimal reproducible example**: Create simplest code that shows the issue

### Issue Template

```markdown
## Description
Clear description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.5]
- Aletheion Core: [e.g., 1.0.0]
- NumPy: [e.g., 1.24.0]

## Code to Reproduce
\`\`\`python
import numpy as np
from src import q_metric

# Minimal code that reproduces issue
...
\`\`\`

## Error Output
\`\`\`
Full error traceback
\`\`\`
```

### Submit Issue

Go to: https://github.com/AletheionAGI/aletheion-core/issues/new

---

## Feature Requests

We welcome feature requests! Please provide:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: What other approaches did you consider?
4. **Additional context**: Any relevant background

**Note**: Feature requests are not guaranteed to be implemented. Consider submitting a PR if you can implement it yourself!

---

## Documentation

### Types of Documentation

1. **Code documentation**: Docstrings in source files
2. **API documentation**: `docs/API.md`
3. **Theory documentation**: `docs/THEORY.md`
4. **Examples**: `examples/*.py`
5. **README**: `README.md`

### Updating Documentation

When making changes:
- Update docstrings if function signatures change
- Update API.md if public API changes
- Add examples for new features
- Update README if installation/usage changes

### Building Documentation (Future)

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build Sphinx docs (when available)
cd docs
make html
```

---

## Development Workflow Summary

```bash
# 1. Setup
git clone https://github.com/YOUR_USERNAME/aletheion-core.git
cd aletheion-core
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"

# 2. Create branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit files ...

# 4. Format and check
black src/ tests/
isort src/ tests/
mypy src/

# 5. Test
pytest tests/

# 6. Commit
git add .
git commit -m "Descriptive message"

# 7. Push
git push origin feature/my-feature

# 8. Create PR on GitHub
```

---

## Project Structure

```
aletheion-core/
├── src/                    # Core implementation
│   ├── __init__.py         # Package initialization
│   ├── q_metric.py         # Q metric implementation
│   ├── varo.py             # VARO operator
│   └── epistemic_gate.py   # Epistemic gating
├── tests/                  # Unit tests
│   ├── test_basic.py       # Integration tests
│   ├── test_q_metric.py    # Q metric tests
│   ├── test_varo.py        # VARO tests
│   └── test_epistemic_gate.py  # Gate tests
├── examples/               # Usage examples
│   ├── simple_demo.py      # Basic demonstration
│   └── arc_example.py      # ARC puzzle example
├── docs/                   # Documentation
│   ├── API.md              # API reference
│   ├── THEORY.md           # Mathematical theory
│   └── CONTRIBUTING.md     # This file
├── .github/
│   └── workflows/
│       └── tests.yml       # CI/CD pipeline
├── README.md               # Project overview
├── LICENSE                 # AGPL-3.0 license
├── setup.py                # Package setup
├── pyproject.toml          # Modern Python packaging
├── requirements.txt        # Runtime dependencies
├── MANIFEST.in             # Distribution files
├── CHANGELOG.md            # Version history
└── CITATION.cff            # Citation information
```

---

## Getting Help

- **Issues**: https://github.com/AletheionAGI/aletheion-core/issues
- **Discussions**: https://github.com/AletheionAGI/aletheion-core/discussions
- **Paper**: https://doi.org/10.13140/RG.2.2.29925.87527
- **Email**: contact@alethea.tech (for commercial inquiries)

---

## License

By contributing to Aletheion Core, you agree that your contributions will be licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See [LICENSE](../LICENSE) for full license text.

For commercial licensing inquiries, contact: licensing@aletheiaengine.dev

---

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md (for each release)
- GitHub contributors page
- Future publications (for significant contributions)

---

Thank you for contributing to Aletheion Core! 🚀
