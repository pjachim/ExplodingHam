# Architecture

This document describes the design philosophy and architecture of ExplodingHam.

## Design Philosophy

ExplodingHam is built around several core principles:

### 1. sklearn API Compatibility

All models follow scikit-learn conventions:
- `fit(X, y)` trains the model and returns `self`
- `predict(X)` makes predictions on new data
- Parameters set in `__init__()`, state stored with trailing underscore

This makes ExplodingHam models drop-in replacements in sklearn pipelines.

### 2. DataFrame Agnostic

We use [Narwhals](https://narwhals-dev.github.io/narwhals/) to support both Pandas and Polars DataFrames transparently:

```python
import narwhals as nw

# Convert input to Narwhals
X = nw.from_native(X, allow_series=True)

# Perform operations
result = X.with_columns(...)

# Convert back to native format
return result.to_native()
```

This means users can use whichever DataFrame library they prefer.

### 3. Dependency Injection

Instead of hardcoding implementations, we accept functions/objects as parameters:

```python
# ❌ Bad: Hardcoded
class Classifier:
    def __init__(self):
        self.compressor = gzip.compress  # Fixed!

# ✅ Good: Injected
class Classifier:
    def __init__(self, compressor):
        self.compressor = compressor  # Flexible!
```

This enables:
- Custom implementations
- Easy testing with mocks
- Experimentation with alternatives

### 4. Minimal Dependencies

We keep external dependencies minimal:
- **Narwhals**: DataFrame compatibility
- **NumPy**: Numerical operations (transitive via Narwhals)
- **Python stdlib**: Everything else

No deep learning frameworks, no heavy ML libraries.

## Package Structure

```
src/explodingham/
├── __init__.py              # Package version, top-level imports
├── models/                  # ML algorithms
│   ├── __init__.py
│   ├── compression_learning/  # Compression-based algorithms
│   │   ├── __init__.py
│   │   └── knn.py          # CompressionKNN, BaseKNNModel
│   └── baseline_models/    # Non-ML baselines
│       ├── __init__.py
│       └── regex.py        # BinaryRegexClassifier
└── utils/                  # Utilities and base classes
    ├── __init__.py
    ├── base/               # Base classes for all models
    │   ├── __init__.py
    │   └── base_classifier.py  # BaseExplodingHamClassifier
    └── distance_metrics/   # Distance/similarity functions
        ├── __init__.py
        └── ncd.py          # NormalizedCompressionDistance
```

### When to Add New Directories

**Create `models/<new_category>/` when:**
- Adding a family of related algorithms
- Examples: `ensemble_methods/`, `neural_approaches/`, `tree_based/`

**Create `utils/<new_category>/` when:**
- Adding utility functions/classes
- Examples: `preprocessing/`, `evaluation/`, `visualization/`

## Class Hierarchy

```
BaseExplodingHamClassifier (from sklearn.base)
│
├── BaseKNNModel
│   └── CompressionKNN
│
└── BinaryRegexClassifier
```

### BaseExplodingHamClassifier

All ExplodingHam classifiers inherit from this base:

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class BaseExplodingHamClassifier(BaseEstimator, ClassifierMixin):
    """Base for all ExplodingHam classifiers."""
    
    def _set_param(self, param, value, type_conversion_func):
        """Helper for parameter setting with type conversion."""
        ...
```

Provides:
- sklearn compatibility (via inheritance)
- Parameter management utilities
- Common interface expectations

### BaseKNNModel

Abstract base for KNN variants with custom distances:

```python
class BaseKNNModel(BaseExplodingHamClassifier):
    """KNN with custom distance metrics."""
    
    def compute_knn(self, a, b, distance_expression, return_predictions=True):
        """Core KNN computation using cross-join and ranking."""
        ...
```

Provides:
- Cross-join logic for pairwise distances
- k-nearest neighbor selection
- Aggregation for predictions

### Concrete Models

Concrete models implement specific algorithms:
- `CompressionKNN`: Uses NCD for distance
- `BinaryRegexClassifier`: Pattern matching

## Data Flow

### Training (fit)

```
Input: X (DataFrame), y (Series)
    ↓
Convert to Narwhals
    ↓
Store/preprocess data
    ↓
Update internal state
    ↓
Return self
```

### Prediction (predict)

```
Input: X (DataFrame)
    ↓
Convert to Narwhals
    ↓
Compute predictions using fitted state
    ↓
Convert back to native format
    ↓
Return predictions
```

## CompressionKNN Architecture

Deep dive into the most complex model:

### 1. Initialization

```python
def __init__(self, k, data_column, encoding, compressor, ...):
    # Resolve compressor (string → function)
    if isinstance(compressor, str):
        self.compressor = self._get_callable_compressor(compressor)
    else:
        self.compressor = compressor
    
    # Store parameters
    self.k = k
    self.data_column = data_column
    ...
```

### 2. Fit Phase

```python
def fit(self, X_train, y_train):
    # Convert to Narwhals
    X = nw.from_native(X_train)
    y = nw.from_native(y_train)
    
    # Precompute compressed lengths (optimization)
    X = X.with_columns(
        compressed_len = X[data_column].map(self._compress)
    )
    
    # Store training data + labels + compressed lengths
    self.model_data = nw.concat([X, y], how='horizontal')
    
    return self
```

### 3. Predict Phase

```python
def predict(self, X_test):
    # Convert to Narwhals
    X = nw.from_native(X_test)
    
    # Build NCD distance expression
    distance_expr = (
        concat_compressed - min(a_compressed, b_compressed)
    ) / max(a_compressed, b_compressed)
    
    # Use base class to find k nearest neighbors
    predictions = self.compute_knn(
        X, 
        self.model_data,
        distance_expr,
        return_predictions=True
    )
    
    return predictions.to_native()
```

### 4. NCD Computation

The tricky part is computing NCD efficiently:

```python
# Cross join test and train data
joined = test.join(train, how='cross')

# For each pair, compute:
# - C(x): already stored in test
# - C(y): already stored in train
# - C(xy): compress(concat(x, y))

joined = joined.with_columns(
    concat_compressed = (col('x') + col('y')).map(compress)
)

# Compute NCD formula
ncd = (concat_compressed - min(C_x, C_y)) / max(C_x, C_y)

# Rank by NCD, keep k smallest
top_k = joined.rank().over('test_id').filter(rank <= k)

# Aggregate: mode of k neighbors' labels
predictions = top_k.group_by('test_id').agg(
    col('label').mode()
)
```

## Testing Strategy

### Test Organization

```
test/
├── test_<module>.py          # Unit tests for single modules
├── test_models/
│   ├── test_compression_learning/
│   │   └── test_knn.py       # Tests for CompressionKNN
│   └── test_baseline_models/
│       └── test_regex.py     # Tests for BinaryRegexClassifier
└── test_utils/
    ├── test_base/
    │   └── test_base_classifier.py
    └── test_distance_metrics/
        └── test_ncd.py       # Tests for NCD
```

### Test Structure

```python
class TestMyModel:
    """Tests for normal usage."""
    
    @pytest.fixture
    def sample_data(self):
        """Reusable test data."""
        ...
    
    def test_initialization(self) -> None:
        """Test object creation."""
        ...
    
    def test_fit(self, sample_data) -> None:
        """Test training."""
        ...
    
    def test_predict(self, sample_data) -> None:
        """Test prediction."""
        ...

class TestEdgeCases:
    """Tests for edge cases and errors."""
    
    def test_empty_data(self) -> None:
        ...
    
    def test_invalid_input(self) -> None:
        ...
```

## Performance Considerations

### CompressionKNN

**Time Complexity:**
- Training: O(n) where n = training samples
- Prediction: O(n × m × c) where m = test samples, c = compression time

**Space Complexity:**
- O(n) - stores all training data + compressed lengths

**Optimization Strategies:**
1. Precompute compressed lengths during fit
2. Use fastest compressor (gzip) by default
3. Support sampling for large datasets

### Future Optimizations

- Parallel compression using multiprocessing
- Incremental learning (partial_fit)
- Approximate nearest neighbors for large datasets
- Caching of compression results

## Extension Points

### Adding New Models

1. **Inherit from appropriate base:**
   ```python
   class MyModel(BaseExplodingHamClassifier):
       ...
   ```

2. **Implement required methods:**
   ```python
   def fit(self, X, y):
       ...
       return self
   
   def predict(self, X):
       ...
       return predictions
   ```

3. **Use Narwhals for DataFrame compatibility:**
   ```python
   X = nw.from_native(X)
   # ... operations ...
   return result.to_native()
   ```

### Adding New Utilities

1. **Create in appropriate utils subdirectory**
2. **Make it reusable and generic**
3. **Add comprehensive docstrings**
4. **Include examples in docstring**

## Design Patterns Used

### 1. Strategy Pattern

Compressor selection in CompressionKNN:

```python
# User provides strategy (compressor)
clf = CompressionKNN(compressor=gzip.compress)

# Or
clf = CompressionKNN(compressor='lzma')  # Resolved internally
```

### 2. Template Method

BaseKNNModel defines the algorithm structure, subclasses fill in specifics:

```python
class BaseKNNModel:
    def compute_knn(self, ...):
        # Template: cross join → distance → rank → filter → aggregate
        ...

class CompressionKNN(BaseKNNModel):
    def predict(self, X):
        # Specific: define distance expression for NCD
        distance = <NCD formula>
        return self.compute_knn(..., distance, ...)
```

### 3. Adapter Pattern

Narwhals adapts different DataFrame libraries to common interface:

```python
# Pandas or Polars input
X_input = pd.DataFrame(...) or pl.DataFrame(...)

# Adapt to common interface
X = nw.from_native(X_input)

# Use common operations
result = X.with_columns(...)

# Adapt back
return result.to_native()
```

## Future Architecture Plans

### Planned Additions

1. **More algorithms**: Rotation Forests, Bumping Classifier, InfoGain Trees
2. **Clustering**: NCD-based clustering (hierarchical, KMeans, DBSCAN)
3. **LLM integration**: Zero-shot and few-shot LLM classifiers
4. **Utilities**: More distance metrics, preprocessing helpers

### Maintaining Consistency

As the library grows:
- All models inherit from `BaseExplodingHamClassifier`
- Use Narwhals for DataFrame compatibility
- Follow sklearn conventions
- Comprehensive docstrings and tests
- Minimal dependencies

## See Also

- [Contributing Guide](contributing.md) - How to contribute
- [API Reference](api/index.md) - Complete API documentation
- [Getting Started](getting_started.md) - First steps with ExplodingHam
