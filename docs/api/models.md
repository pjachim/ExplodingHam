# Models API Reference

Complete reference for all ExplodingHam models.

## Overview

ExplodingHam provides two categories of models:

1. **[Compression Learning](#compression-learning)** - Novel ML algorithms based on compression theory
2. **[Baseline Models](#baseline-models)** - Non-ML baselines for quick comparisons

All models follow sklearn conventions and are compatible with Pandas and Polars DataFrames.

## Compression Learning

Models that use compression theory and information theory for machine learning.

### CompressionKNN

K-Nearest Neighbors using Normalized Compression Distance (NCD).

```python
from explodingham.models.compression_learning.knn import CompressionKNN

clf = CompressionKNN(k=3, data_column='text', compressor='gzip')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

**Key Features:**
- 📝 No feature engineering required
- 🔬 Based on Kolmogorov complexity
- 🌍 Universal similarity metric
- 📊 Works with text, DNA, sequences

**[Full Documentation →](compression_learning.md#compressionknn)**

### BaseKNNModel

Abstract base class for KNN with custom distance metrics.

```python
from explodingham.models.compression_learning.knn import BaseKNNModel

class MyKNN(BaseKNNModel):
    # Implement custom distance metric
    ...
```

**[Full Documentation →](compression_learning.md#baseknnmodel)**

## Baseline Models

Non-ML classifiers for establishing baselines and implementing known rules.

### BinaryRegexClassifier

Pattern matching classifier using regular expressions.

```python
from explodingham.models.baseline_models.regex import BinaryRegexClassifier

clf = BinaryRegexClassifier(
    pattern=r'https?://',
    match_type='partial',
    match_prediction='URL',
    no_match_prediction='Not URL'
)
predictions = clf.predict(data)
```

**Key Features:**
- ⚡ Fast pattern matching
- 🎯 Sklearn-compatible
- 🛠️ Great for baselines
- 🔍 Flexible matching options

**[Full Documentation →](baseline_models.md#binaryregexclassifier)**

## Comparison Table

| Feature | CompressionKNN | BinaryRegexClassifier |
|---------|----------------|----------------------|
| **Type** | ML (learns from data) | Non-ML (rule-based) |
| **Speed** | 🐢 Slow | ⚡ Very fast |
| **Feature Engineering** | ❌ None | ❌ None |
| **Pattern Detection** | ✅ Automatic | ⚠️ Manual (regex) |
| **Best For** | Text, sequences, novel data | Known patterns, baselines |
| **Accuracy** | 🎯 Can be high | ⚠️ Depends on rules |

## Usage Patterns

### Standard Workflow

```python
# 1. Import model
from explodingham.models.compression_learning.knn import CompressionKNN

# 2. Initialize with parameters
clf = CompressionKNN(k=3, data_column='text')

# 3. Train on data
clf.fit(X_train, y_train)

# 4. Make predictions
predictions = clf.predict(X_test)
```

### With sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Models work in sklearn pipelines
pipeline = Pipeline([
    ('clf', CompressionKNN(k=3, data_column='text'))
])

# Cross-validation
# scores = cross_val_score(pipeline, X, y, cv=5)
```

### With Polars

```python
import polars as pl

# Works seamlessly with Polars
X_train_pl = pl.DataFrame({'text': ['...']})
y_train_pl = pl.Series('label', ['...'])

clf = CompressionKNN(k=2, data_column='text')
clf.fit(X_train_pl, y_train_pl)
```

## Common Parameters

### All Models

All ExplodingHam models support:
- **DataFrame compatibility**: Pandas and Polars
- **sklearn API**: `fit()`, `predict()` methods
- **Method chaining**: `fit()` returns `self`

### Compression Models

- **`compressor`**: 'gzip', 'bz2', 'lzma', or custom function
- **`encoding`**: String encoding (default: 'utf-8')
- **`data_column`**: Which column to analyze

### Regex Models

- **`pattern`**: Regular expression pattern
- **`match_type`**: 'partial' or 'full'
- **`ignore_case`**: Case-insensitive matching
- **`match_prediction`**: Value for matches
- **`no_match_prediction`**: Value for non-matches

## Model Selection Guide

### Choose CompressionKNN when:
- ✅ You have text or sequential data
- ✅ You want to avoid feature engineering
- ✅ Dataset is small-to-medium (<10K samples)
- ✅ Patterns are more important than individual words

### Choose BinaryRegexClassifier when:
- ✅ You need a quick baseline
- ✅ You have clear patterns (emails, URLs, IDs)
- ✅ Speed is critical
- ✅ You want to test ML against simple rules

## Related Documentation

### Tutorials
- [Compression KNN Tutorial](../../tutorials/compression_knn.md) - Deep dive
- [Regex Classifier Tutorial](../../tutorials/regex_classifier.md) - Pattern matching

### Examples
- [Text Classification](../../examples/text_classification.md) - Language detection, sentiment
- [DNA Sequences](../../examples/dna_sequences.md) - Bioinformatics
- [Custom Compressors](../../examples/custom_compressors.md) - Build your own

### Utilities
- [Distance Metrics](../utils/distance_metrics.md) - NCD and other metrics
- [Base Classes](../utils/base.md) - Extending ExplodingHam

## Next Steps

1. **[Get Started](../../getting_started.md)** - Quick start guide
2. **[Choose a Tutorial](../../tutorials/compression_knn.md)** - Learn in depth
3. **[Browse Examples](../../examples/text_classification.md)** - See real applications
4. **[Read API Docs](compression_learning.md)** - Complete reference

---

**Need help?** Check the [Getting Started guide](../../getting_started.md) or [open an issue](https://github.com/pjachim/ExplodingHam/issues).
