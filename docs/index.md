# ExplodingHam Documentation

**Version:** 0.0.1

> *Oddball scikit-learn style ML+ approaches.*

## Welcome

ExplodingHam is a Python library that provides scikit-learn-compatible implementations of obscure, experimental, and unconventional machine learning algorithms. Our goal is to extend beyond what scikit-learn offers, including both novel ML approaches and non-ML baseline methods that integrate seamlessly with your existing sklearn workflows.

## Why ExplodingHam?

- **🎯 sklearn API Compatibility**: All models follow scikit-learn conventions (`fit()`, `predict()`, etc.), making integration effortless
- **🔬 Experimental Algorithms**: Access to cutting-edge and lesser-known ML approaches not available in mainstream libraries
- **📊 DataFrame Agnostic**: Built with [Narwhals](https://narwhals-dev.github.io/narwhals/), works seamlessly with Pandas, Polars, and other dataframe libraries
- **🛠️ Research-Friendly**: Perfect for academic research, experimentation, and exploring unconventional approaches
- **⚡ Easy Baselines**: Includes non-ML baseline classifiers (like RegEx) to quickly establish performance benchmarks

## Quick Start

```python
# Install
pip install explodingham

# Use compression-based KNN for text classification
from explodingham.models.compression_learning.knn import CompressionKNN
import pandas as pd

# Prepare data
X_train = pd.DataFrame({
    'text': ['Hello world', 'Bonjour monde', 'Hola mundo']
})
y_train = pd.Series(['English', 'French', 'Spanish'])

# Train and predict
clf = CompressionKNN(k=1, data_column='text')
clf.fit(X_train, y_train)

X_test = pd.DataFrame({'text': ['Hi there']})
predictions = clf.predict(X_test)  # ['English']
```

## Current Models

### Compression Learning
- **[CompressionKNN](api/models/compression_learning.md#compressionknn)** - K-Nearest Neighbors using Normalized Compression Distance (NCD). Perfect for text, DNA sequences, and any sequential data.

### Baseline Models
- **[BinaryRegexClassifier](api/models/baseline_models.md#binaryregexclassifier)** - Quick and dirty regex-based classifier for establishing baselines (not ML, but sklearn-compatible).

### Utilities
- **[NormalizedCompressionDistance](api/utils/distance_metrics.md#normalizedcompressiondistance)** - Universal similarity metric based on Kolmogorov complexity.

## Planned Features

The following algorithms are planned for future releases:

- **Bumping Classifier** - Ensemble method that trains multiple models on data subsets and keeps the best performer
- **Rotation Forests** - Random forest variant that performs PCA before each tree
- **InfoGain Trees** - ID3/C4.5 classifiers on non-binary trees
- **Zero-Shot LLM Classifier** - LLM-based classifier using prompts
- **Few-Shot LLM Classifier** - LLM classifier with example-based learning
- **Compression-based Clustering** - Hierarchical, KMeans, and DBSCAN using compression distances

## Documentation Structure

### Getting Started
- **[Installation](installation.md)** - Installation instructions and requirements
- **[Getting Started Guide](getting_started.md)** - Your first steps with ExplodingHam

### Tutorials
- **[Compression KNN Tutorial](tutorials/compression_knn.md)** - In-depth guide to compression-based classification
- **[Regex Classifier Tutorial](tutorials/regex_classifier.md)** - Using regex patterns for quick baselines

### Examples
- **[Text Classification](examples/text_classification.md)** - Language detection and sentiment analysis
- **[DNA Sequence Classification](examples/dna_sequences.md)** - Bioinformatics applications
- **[Custom Compressors](examples/custom_compressors.md)** - Building custom compression functions

### API Reference
- **[Models API](api/models.md)** - Complete model reference
  - [Compression Learning](api/models/compression_learning.md)
  - [Baseline Models](api/models/baseline_models.md)
- **[Utils API](api/utils.md)** - Utilities and helpers
  - [Distance Metrics](api/utils/distance_metrics.md)
  - [Base Classes](api/utils/base.md)

### Development
- **[Contributing](contributing.md)** - How to contribute to ExplodingHam
- **[Architecture](architecture.md)** - Design philosophy and project structure

## Philosophy

ExplodingHam focuses on algorithms that:

1. **Didn't make it into sklearn** - Either too experimental, niche, or unconventional
2. **Solve real problems** - Not just academic curiosities
3. **Follow sklearn patterns** - Seamless integration with existing workflows

## Community

- **GitHub**: [https://github.com/pjachim/ExplodingHam](https://github.com/pjachim/ExplodingHam)
- **Issues**: [Report bugs or request features](https://github.com/pjachim/ExplodingHam/issues)
- **PyPI**: [https://pypi.org/project/explodingham/](https://pypi.org/project/explodingham/)

## License

ExplodingHam is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

---

**Next Steps**: Start with the [Installation Guide](installation.md) or jump straight into the [Getting Started Guide](getting_started.md)!
