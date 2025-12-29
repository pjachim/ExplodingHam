# Utils API Reference

Utilities and helper functions for ExplodingHam.

## Overview

The utilities module provides:

1. **[Distance Metrics](#distance-metrics)** - Similarity and distance measures
2. **[Base Classes](#base-classes)** - Foundation for all models

## Distance Metrics

### NormalizedCompressionDistance

Universal similarity metric based on compression.

```python
from explodingham.utils.distance_metrics.ncd import NormalizedCompressionDistance
import gzip

ncd = NormalizedCompressionDistance(gzip.compress)
distance = ncd.ncd(b"hello world", b"hello there")
# Returns: ~0.45
```

**[Full Documentation →](distance_metrics.md#normalizedcompressiondistance)**

## Base Classes

### BaseExplodingHamClassifier

Base class for all ExplodingHam classifiers.

```python
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier

class MyClassifier(BaseExplodingHamClassifier):
    def __init__(self, param):
        self._set_param('param', param, int)
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return predictions
```

**[Full Documentation →](base.md#baseexplodinghamclassifier)**

## Quick Reference

### Distance Metrics

| Metric | Input | Output | Use Case |
|--------|-------|--------|----------|
| NormalizedCompressionDistance | bytes | float (0-1) | Text/sequence similarity |

### Base Classes

| Class | Purpose | Inherit When |
|-------|---------|--------------|
| BaseExplodingHamClassifier | Classifier foundation | Creating any classifier |
| BaseKNNModel | KNN with custom distance | Creating KNN variant |

## See Also

- [Models API](models.md) - Available models
- [Architecture](../../architecture.md) - Design overview
- [Contributing](../../contributing.md) - Add new utilities
