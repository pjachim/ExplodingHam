# utils.base

Base classes for ExplodingHam models.

## BaseExplodingHamClassifier

```python
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
```

Base classifier class for all ExplodingHam models.

### Description

Abstract base class providing common functionality for all classifiers in ExplodingHam. Ensures consistency across the library and provides utility methods for parameter management.

### Methods

#### _set_param

```python
_set_param(name: str, value: Any, type_cast: Callable) -> None
```

Set a parameter with optional type casting.

**Parameters:**
- `name` (str): Parameter name to set as attribute
- `value` (Any): Value to set
- `type_cast` (Callable): Function to apply to value before setting

**Example:**
```python
class MyClassifier(BaseExplodingHamClassifier):
    def __init__(self, threshold):
        self._set_param('threshold', threshold, float)
        # Sets self.threshold = float(threshold)
```

### Usage

All ExplodingHam classifiers inherit from this base class:

```python
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier

class CustomClassifier(BaseExplodingHamClassifier):
    """Your custom classifier."""
    
    def __init__(self, param1, param2):
        # Use base class utilities
        self._set_param('param1', param1, str)
        self._set_param('param2', param2, int)
    
    def fit(self, X, y):
        """Fit method - implement your training logic."""
        # Your implementation
        return self
    
    def predict(self, X):
        """Predict method - implement your prediction logic."""
        # Your implementation
        return predictions
```

### Design Philosophy

The base classifier:
- **Minimal interface**: Doesn't impose unnecessary requirements
- **Utility methods**: Provides helpful parameter management
- **Extensible**: Easy to subclass for new models
- **Convention-based**: Follows sklearn patterns

### Relationship to sklearn

ExplodingHam models follow sklearn conventions but don't strictly inherit from sklearn base classes. This provides:
- **Flexibility**: Freedom to use different DataFrame libraries (via Narwhals)
- **Compatibility**: Models still work with sklearn utilities (pipelines, cross-validation)
- **Simplicity**: Less coupling to sklearn internals

### Current Subclasses

All ExplodingHam classifiers inherit from `BaseExplodingHamClassifier`:

- [CompressionKNN](../models/compression_learning.md#compressionknn)
- [BaseKNNModel](../models/compression_learning.md#baseknnmodel)
- [BinaryRegexClassifier](../models/baseline_models.md#binaryregexclassifier)

### Example: Creating a New Classifier

```python
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import narwhals as nw

class SimpleThresholdClassifier(BaseExplodingHamClassifier):
    """
    Classify based on a simple threshold.
    
    Parameters
    ----------
    threshold : float
        Values above threshold are class 1, below are class 0
    column : str
        Column to apply threshold to
    """
    
    def __init__(self, threshold: float, column: str):
        self._set_param('threshold', threshold, float)
        self._set_param('column', column, str)
    
    def fit(self, X, y):
        """Fit is a no-op for this simple classifier."""
        return self
    
    def predict(self, X):
        """Predict based on threshold."""
        X = nw.from_native(X)
        
        predictions = nw.when(nw.col(self.column) > self.threshold).then(1).otherwise(0)
        
        return X.with_columns(predictions.alias('prediction')).to_native()

# Use it
clf = SimpleThresholdClassifier(threshold=5.0, column='value')
clf.fit(None, None)  # No training needed
predictions = clf.predict(df)
```

### Best Practices

When creating new classifiers:

1. **Inherit from BaseExplodingHamClassifier**
   ```python
   class MyClassifier(BaseExplodingHamClassifier):
       pass
   ```

2. **Use `_set_param()` for parameter setting**
   ```python
   def __init__(self, param):
       self._set_param('param', param, int)
   ```

3. **Follow sklearn conventions**
   - Implement `fit(X, y)` that returns `self`
   - Implement `predict(X)` that returns predictions
   - Store fitted state as attributes ending with `_`

4. **Use Narwhals for DataFrame compatibility**
   ```python
   X = nw.from_native(X)  # Works with pandas/polars
   ```

5. **Provide comprehensive docstrings**
   - Use NumPy-style docstrings
   - Include Parameters, Returns, Examples sections
   - Explain the theoretical basis

### See Also

- [CompressionKNN](../models/compression_learning.md#compressionknn) - Example subclass
- [BinaryRegexClassifier](../models/baseline_models.md#binaryregexclassifier) - Example subclass
- [Contributing Guide](../../contributing.md) - How to add new models

---

The base classifier provides a minimal but helpful foundation for all ExplodingHam models. Subclass it when creating new classifiers to maintain consistency across the library.
