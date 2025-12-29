# ExplodingHam AI Agent Instructions

## Project Overview
ExplodingHam is a Python library for oddball machine learning algorithms that didn't make it into scikit-learn. The goal is to mirror sklearn's API patterns while implementing obscure or experimental algorithms.

## Architecture

### Package Structure
- **`src/explodingham/`** - Main package source (note: package name is `explodingham`, not `explodingham`)
  - **`models/`** - ML algorithm implementations (organized by algorithm family)
    - `compression_learning/` - Algorithms based on compression theory
  - **`utils/`** - Utility functions and helper classes
    - `distance_metrics/` - Distance/similarity metrics (e.g., `ncd.py` for Normalized Compression Distance)
  - **`test/`** - Test suite (currently minimal, needs expansion)

### Design Philosophy
- **sklearn API compatibility**: Classes should follow sklearn conventions (e.g., `fit()`, `predict()`, `transform()` methods)
- **Dependency injection pattern**: Utilities like `NormalizedCompressionDistance` accept compressor functions as parameters, allowing flexibility without hardcoding implementations
- **Pure Python**: No external dependencies beyond standard library where possible (check `pyproject.toml` before adding deps)

## Development Workflow

### Setup
```bash
# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Unix

# Install in editable mode with test dependencies
pip install -e .
pip install pytest  # Required for running tests
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_ncd.py

# Run with verbose output
pytest -v

# Run specific test class
pytest test/test_ncd.py::TestNormalizedCompressionDistance
```

### Version Management
- Version is defined in **both** `pyproject.toml` AND `src/explodingham/__init__.py`
- **CRITICAL**: Keep these synchronized when bumping versions
- Current version: 0.0.1

### Publishing
- Auto-publishes to PyPI on push to `main` branch via GitHub Actions
- Workflow: `.github/workflows/python-publish.yml`
- Uses trusted publishing (no manual token management)

## Code Conventions

### Class Design Pattern (see `ncd.py`)
```python
class AlgorithmName:
    """Clear docstring explaining the algorithm and its theoretical basis."""
    
    def __init__(self, configurable_dependency):
        """Accept dependencies/config via constructor (dependency injection)."""
        self.dependency = configurable_dependency
    
    def core_method(self, x, y):
        """Main algorithm logic with descriptive parameter names."""
        # Implementation
```

### File Organization
- New algorithms go in `models/<algorithm_family>/`
- New metrics/utilities go in `utils/<category>/`
- Always include `__init__.py` in new directories (even if empty)
- Module files should be lowercase with underscores (e.g., `ncd.py`)

### Documentation Style
- **Use NumPy-style docstrings** with sections: Parameters, Returns, Examples, Notes
- Include extensive **Examples** section with executable doctest examples
- Explain the **why** (theoretical basis) not just the **what** in class/module docstrings
- Use mathematical notation in Notes section when appropriate (e.g., LaTeX via Sphinx)
- Example: `NormalizedCompressionDistance` docstring mentions Kolmogorov complexity and includes formula

### Type Hints
- **All function signatures must include type hints** for parameters and return values
- Use modern Python typing syntax: `str | bytes` instead of `Union[str, bytes]`
- Import types from `typing` module at top of file
- Example: `def ncd(self, x: str | bytes, y: str | bytes) -> float:`

## Common Tasks

### Adding a New Algorithm
1. Determine if it's a model (goes in `models/`) or utility (goes in `utils/`)
2. Create subdirectory if it represents a new algorithm family
3. Implement class following sklearn API patterns
4. Add to relevant `__init__.py` for imports
5. Test with `pip install -e .` to verify package structure

### Testing
- **Test framework**: pytest with class-based test organization
- **Test location**: `test/` directory with `test_*.py` naming convention
- **Test structure**: Group related tests in classes (e.g., `TestNormalizedCompressionDistance`)
- Use pytest fixtures for setup (e.g., `@pytest.fixture` for creating test instances)
- Include multiple test classes per file for different scenarios (e.g., `TestEdgeCases`, `TestIntegration`)
- Add type hints to test methods: `-> None` for test functions
- Test file naming: `test_<module_name>.py` (e.g., `test_ncd.py` for `ncd.py`)

## Project Status
**Under active development** - Many algorithm slots are planned but not yet implemented. Refer to README.md for the project vision. When implementing new algorithms, prioritize lesser-known algorithms that complement (rather than duplicate) scikit-learn's offerings.

## Key Files to Review
- `pyproject.toml` - Package metadata, dependencies, build configuration
- `src/explodingham/__init__.py` - Package version and top-level imports
- `src/explodingham/utils/distance_metrics/ncd.py` - Reference implementation showing:
  - NumPy-style docstrings with extensive examples
  - Type hints using modern syntax
  - Dependency injection pattern
- `test/test_ncd.py` - Reference test file showing:
  - Pytest class-based test organization
  - Use of fixtures for setup
  - Multiple test classes for different scenarios
  - Type hints in test methods
