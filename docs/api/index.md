# API Reference

Complete API documentation for ExplodingHam's classes and functions.

## Models

### Compression Learning
- [CompressionKNN](models/compression_learning.md#compressionknn) - K-Nearest Neighbors with Normalized Compression Distance
- [BaseKNNModel](models/compression_learning.md#baseknnmodel) - Base class for KNN models with custom distance metrics

### Baseline Models
- [BinaryRegexClassifier](models/baseline_models.md#binaryregexclassifier) - Regex-based binary classifier

## Utilities

### Distance Metrics
- [NormalizedCompressionDistance](utils/distance_metrics.md#normalizedcompressiondistance) - NCD metric for similarity measurement

### Base Classes
- [BaseExplodingHamClassifier](utils/base.md#baseexplodinghamclassifier) - Base classifier for all ExplodingHam models

## Quick Links

### By Task
- **Text Classification**: [CompressionKNN](models/compression_learning.md#compressionknn), [BinaryRegexClassifier](models/baseline_models.md#binaryregexclassifier)
- **Sequence Analysis**: [CompressionKNN](models/compression_learning.md#compressionknn), [NormalizedCompressionDistance](utils/distance_metrics.md#normalizedcompressiondistance)
- **Quick Baselines**: [BinaryRegexClassifier](models/baseline_models.md#binaryregexclassifier)

### By Category
- **Classifiers**: All models inherit from [BaseExplodingHamClassifier](utils/base.md#baseexplodinghamclassifier)
- **Distance Metrics**: [NormalizedCompressionDistance](utils/distance_metrics.md#normalizedcompressiondistance)
- **Base Classes**: [BaseKNNModel](models/compression_learning.md#baseknnmodel), [BaseExplodingHamClassifier](utils/base.md#baseexplodinghamclassifier)

## Index

- [models.compression_learning](models/compression_learning.md)
  - CompressionKNN
  - BaseKNNModel
- [models.baseline_models](models/baseline_models.md)
  - BinaryRegexClassifier
- [utils.distance_metrics](utils/distance_metrics.md)
  - NormalizedCompressionDistance
- [utils.base](utils/base.md)
  - BaseExplodingHamClassifier
