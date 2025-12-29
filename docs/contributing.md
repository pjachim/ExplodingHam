# Contributing to ExplodingHam

Thank you for your interest in contributing to ExplodingHam! This guide will help you get started.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Coding Standards](#coding-standards)
4. [Adding New Algorithms](#adding-new-algorithms)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Setup Steps

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ExplodingHam.git
cd ExplodingHam

# 3. Create a virtual environment
python -m venv .venv

# 4. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate

# 5. Install in editable mode
pip install -e .

# 6. Install development dependencies
pip install pytest

# 7. Create a feature branch
git checkout -b feature/your-feature-name
```

### Verify Installation

```python
import explodingham
print(explodingham.__version__)  # Should print: 0.0.1

# Try importing models
from explodingham.models.compression_learning.knn import CompressionKNN
print("✓ Setup successful!")
```

## Project Structure

```
ExplodingHam/
├── src/explodingham/          # Main package source
│   ├── __init__.py            # Package version and imports
│   ├── models/                # ML algorithm implementations
│   │   ├── compression_learning/  # Compression-based algorithms
│   │   │   ├── __init__.py
│   │   │   └── knn.py         # CompressionKNN
│   │   └── baseline_models/   # Non-ML baseline classifiers
│   │       ├── __init__.py
│   │       └── regex.py       # BinaryRegexClassifier
│   └── utils/                 # Utility functions and base classes
│       ├── distance_metrics/  # Distance/similarity metrics
│       │   ├── __init__.py
│       │   └── ncd.py         # NormalizedCompressionDistance
│       └── base/              # Base classes
│           ├── __init__.py
│           └── base_classifier.py  # BaseExplodingHamClassifier
├── test/                      # Test suite
│   ├── __init__.py
│   ├── test_ncd.py
│   ├── test_models/
│   │   ├── test_compression_learning/
│   │   └── test_baseline_models/
│   └── test_utils/
├── docs/                      # Documentation
│   ├── index.md               # Documentation homepage
│   ├── tutorials/             # In-depth guides
│   ├── examples/              # Example use cases
│   └── api/                   # API reference
├── .scratchpads/              # Jupyter notebooks for testing/demos
├── pyproject.toml             # Package metadata and dependencies
└── README.md                  # Project overview
```

## Coding Standards

### Design Philosophy

1. **sklearn API compatibility**: Follow scikit-learn conventions (`fit()`, `predict()`, etc.)
2. **Dependency injection**: Accept dependencies via constructor (e.g., compressor functions)
3. **DataFrame agnostic**: Use Narwhals for Pandas/Polars compatibility
4. **Pure Python**: Minimize external dependencies

### Code Style

#### Type Hints

**All function signatures must include type hints:**

```python
# ✅ Good
def ncd(self, x: str | bytes, y: str | bytes) -> float:
    """Compute NCD between x and y."""
    ...

# ❌ Bad
def ncd(self, x, y):
    """Compute NCD between x and y."""
    ...
```

Use modern Python typing syntax:
```python
# ✅ Good
str | bytes  # Python 3.10+

# ❌ Bad
Union[str, bytes]  # Old style
```

#### Docstrings

**Use NumPy-style docstrings** with extensive examples:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of what the function does.
    
    More detailed explanation if needed. Explain the theoretical
    basis or algorithm if applicable.
    
    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.
    
    Returns
    -------
    bool
        Description of return value.
    
    Examples
    --------
    >>> result = my_function(5, "test")
    >>> print(result)
    True
    
    Notes
    -----
    Additional notes about implementation details, performance
    characteristics, or theoretical background.
    
    References
    ----------
    .. [1] Author. "Paper Title." Journal, Year.
    """
    pass
```

#### File Organization

```python
# Imports at top
from typing import Callable
import narwhals as nw

# Class definition
class MyAlgorithm:
    """Class docstring with theory and examples."""
    
    def __init__(self, param: int):
        """Constructor with parameter docs."""
        self.param = param
    
    def fit(self, X, y):
        """Fit method."""
        ...
        return self
    
    def predict(self, X):
        """Predict method."""
        ...
        return predictions
    
    def _private_helper(self):
        """Private methods start with underscore."""
        pass
```

### Naming Conventions

- **Modules**: Lowercase with underscores (`compression_learning`, `ncd.py`)
- **Classes**: PascalCase (`CompressionKNN`, `BinaryRegexClassifier`)
- **Functions/Methods**: Lowercase with underscores (`compute_knn`, `compress_size`)
- **Constants**: UPPERCASE (`REGEX_MAPPING`)
- **Private**: Leading underscore (`_private_method`, `_internal_state`)

## Adding New Algorithms

### Step 1: Determine Category

**Is it a model?** → `src/explodingham/models/<category>/`

**Is it a utility?** → `src/explodingham/utils/<category>/`

### Step 2: Create Module File

```bash
# For a new model
touch src/explodingham/models/my_category/my_algorithm.py

# For a new utility
touch src/explodingham/utils/my_category/my_helper.py
```

### Step 3: Implement the Algorithm

```python
# src/explodingham/models/my_category/my_algorithm.py

from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import narwhals as nw
from typing import Any

class MyAlgorithm(BaseExplodingHamClassifier):
    """
    Brief description of your algorithm.
    
    Detailed explanation of the theory, use cases, and how it works.
    Cite relevant papers if applicable.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, default=value
        Description of param2.
    
    Attributes
    ----------
    param1 : type
        Description.
    param2 : type
        Description.
    _fitted : bool
        Whether the model has been fitted.
    
    Examples
    --------
    >>> from explodingham.models.my_category.my_algorithm import MyAlgorithm
    >>> import pandas as pd
    >>> 
    >>> X = pd.DataFrame({'feature': [1, 2, 3]})
    >>> y = pd.Series(['A', 'B', 'A'])
    >>> 
    >>> clf = MyAlgorithm(param1=5)
    >>> clf.fit(X, y)
    >>> predictions = clf.predict(X)
    
    Notes
    -----
    Explain theoretical foundations, complexity, limitations.
    
    References
    ----------
    .. [1] Author. "Paper Title." Conference/Journal, Year.
    """
    
    def __init__(
        self,
        param1: int,
        param2: str = 'default'
    ):
        self.param1 = param1
        self.param2 = param2
        self._fitted = False
    
    def fit(self, X, y):
        """
        Fit the model on training data.
        
        Parameters
        ----------
        X : DataFrame or Series
            Training features.
        y : Series
            Training labels.
        
        Returns
        -------
        self : MyAlgorithm
            Fitted instance.
        """
        # Convert to Narwhals for DataFrame agnosticism
        X = nw.from_native(X, allow_series=True)
        y = nw.from_native(y, allow_series=True)
        
        # Your fitting logic here
        self._fitted = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : DataFrame or Series
            Test features.
        
        Returns
        -------
        predictions : Series
            Predicted labels.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = nw.from_native(X, allow_series=True)
        
        # Your prediction logic here
        predictions = ...
        
        return predictions.to_native()
```

### Step 4: Update `__init__.py`

```python
# src/explodingham/models/my_category/__init__.py

from .my_algorithm import MyAlgorithm

__all__ = ['MyAlgorithm']
```

### Step 5: Add Tests

```python
# test/test_models/test_my_category/test_my_algorithm.py

import pytest
import pandas as pd
from explodingham.models.my_category.my_algorithm import MyAlgorithm

class TestMyAlgorithm:
    """Tests for MyAlgorithm."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series(['A', 'B', 'A', 'B', 'A'])
        return X, y
    
    def test_initialization(self) -> None:
        """Test algorithm initialization."""
        clf = MyAlgorithm(param1=5)
        assert clf.param1 == 5
        assert clf.param2 == 'default'
    
    def test_fit(self, sample_data) -> None:
        """Test fitting."""
        X, y = sample_data
        clf = MyAlgorithm(param1=5)
        result = clf.fit(X, y)
        assert result is clf  # fit should return self
        assert clf._fitted
    
    def test_predict(self, sample_data) -> None:
        """Test prediction."""
        X, y = sample_data
        clf = MyAlgorithm(param1=5)
        clf.fit(X, y)
        predictions = clf.predict(X)
        assert len(predictions) == len(X)
    
    def test_predict_before_fit(self, sample_data) -> None:
        """Test that predict fails if not fitted."""
        X, _ = sample_data
        clf = MyAlgorithm(param1=5)
        with pytest.raises(ValueError, match="must be fitted"):
            clf.predict(X)

class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_input(self) -> None:
        """Test with empty data."""
        clf = MyAlgorithm(param1=5)
        X = pd.DataFrame({'feature': []})
        y = pd.Series([])
        # Should handle gracefully
        clf.fit(X, y)
```

### Step 6: Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_models/test_my_category/test_my_algorithm.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=explodingham
```

## Testing

### Test Structure

- **Test files**: `test_<module_name>.py`
- **Test classes**: Group related tests (`TestMyAlgorithm`, `TestEdgeCases`)
- **Test methods**: Descriptive names (`test_initialization`, `test_fit_with_polars`)

### Writing Good Tests

```python
def test_specific_behavior(self) -> None:
    """Test description explaining what's being tested."""
    # Arrange: Set up test data
    X = pd.DataFrame({'col': [1, 2, 3]})
    y = pd.Series(['A', 'B', 'A'])
    clf = MyAlgorithm(param=5)
    
    # Act: Perform the operation
    result = clf.fit(X, y)
    
    # Assert: Verify expectations
    assert result is clf
    assert clf._fitted is True
```

### Test Coverage

Aim for comprehensive coverage:
- ✅ Happy path (normal usage)
- ✅ Edge cases (empty data, single sample)
- ✅ Error cases (invalid inputs)
- ✅ Both Pandas and Polars (if applicable)
- ✅ Different parameter combinations

## Documentation

### Required Documentation

For every new feature, provide:

1. **Docstrings** in the code (NumPy style)
2. **API reference** in `docs/api/`
3. **Tutorial** in `docs/tutorials/` (for major features)
4. **Examples** in `docs/examples/`

### API Reference Template

```markdown
## MyAlgorithm

Brief description.

### Class Signature

\`\`\`python
MyAlgorithm(param1: int, param2: str = 'default')
\`\`\`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param1` | int | required | Description |
| `param2` | str | 'default' | Description |

## Submitting Changes

### Before Submitting

1. **Run tests**: `pytest`
2. **Check imports**: Verify all imports work
3. **Update version**: If needed, update version in `pyproject.toml` and `src/explodingham/__init__.py`
4. **Write documentation**: API docs, tutorials, examples
5. **Update CHANGELOG**: Document your changes

### Commit Messages

Follow conventional commit format:

```
feat: add MyAlgorithm for X task

Detailed description of what the feature does and why it's useful.

Closes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding tests
- `refactor`: Code restructuring
- `perf`: Performance improvement

### Pull Request Process

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**
   - Provide clear description
   - Reference related issues
   - Include examples/screenshots if applicable

3. **Address review feedback**
   - Make requested changes
   - Push updates to same branch

4. **Merge**
   - Maintainer will merge once approved

## Version Management

**CRITICAL**: Version must be updated in TWO places:

1. `pyproject.toml`:
   ```toml
   version = "0.0.2"
   ```

2. `src/explodingham/__init__.py`:
   ```python
   __version__ = '0.0.2'
   ```

## Questions?

- **GitHub Issues**: [Report bugs or request features](https://github.com/pjachim/ExplodingHam/issues)
- **Discussions**: Ask questions on GitHub Discussions

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and build cool stuff together!

---

Thank you for contributing to ExplodingHam! 🎉
