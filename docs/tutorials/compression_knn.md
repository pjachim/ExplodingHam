# Compression KNN Tutorial

## Introduction

The `CompressionKNN` leverages compression algorithms to classify data without any feature engineering. This tutorial will show you how to use it effectively.

## Table of Contents

1. [Theory Behind Compression-Based Classification](#theory)
2. [Basic Usage](#basic-usage)
3. [Choosing the Right Compressor](#compressors)
4. [Custom Compression Functions](#custom-compressors)
5. [Working with Different Data Types](#data-types)
6. [Performance Tuning](#performance)
7. [Real-World Applications](#applications)

## <a name="theory"></a>Theory Behind Compression-Based Classification

### The Core Insight

**Similar objects compress well together.**

When you concatenate two similar strings and compress them, the result is often smaller than the sum of their individual compressed sizes. Why? Because compression algorithms exploit patterns and redundancy.

### Normalized Compression Distance

The Normalized Compression Distance (NCD) quantifies similarity:

$$
NCD(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}
$$

where:
- $C(x)$ = compressed size of string $x$
- $xy$ = concatenation of $x$ and $y$

**Properties:**
- NCD ≈ 0 for identical objects
- NCD ≈ 1 for completely different objects
- NCD is approximately symmetric and satisfies triangle inequality
- It's a universal similarity metric (works for any data type)


## <a name="basic-usage"></a>Basic Usage

### Simple Language Classification

```python
from explodingham.models.compression_learning.knn import CompressionKNN
import pandas as pd

# Training data
X_train = pd.DataFrame({
    'text': [
        'Hello world, how are you?',
        'Good morning, nice to meet you',
        'Hola mundo, cómo estás?',
        'Buenos días, mucho gusto',
        'Bonjour le monde, comment allez-vous?',
        'Bonne journée, enchanté',
    ]
})

y_train = pd.Series(['English', 'English', 'Spanish', 'Spanish', 'French', 'French'])

# Test data
X_test = pd.DataFrame({
    'text': [
        'Hi there, what is your name?',
        'Hola amigo, qué tal?',
        'Bonjour, ça va?'
    ]
})

# Create and train
knn = CompressionKNN(k=2, data_column='text')
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)
print("Predictions:", predictions.to_list())
# Output: ['English', 'Spanish', 'French']
```

### Understanding the Parameters

```python
knn = CompressionKNN(
    k=2,                    # Number of neighbors to consider
    data_column='text',     # Column with data to compress
    compressor='gzip',      # Compression algorithm
    encoding='utf-8',       # String encoding
    target_column='target'  # Label column name (for fit)
)
```

## <a name="compressors"></a>Choosing the Right Compressor

ExplodingHam provides three built-in compressors, each with different trade-offs:

### GZIP (Default)

**Best for:** General-purpose use, when speed matters

```python
knn = CompressionKNN(k=2, data_column='text', compressor='gzip')
```

**Characteristics:**
- ⚡ Fast
- 📊 Used in the original paper

### Comparison Example

```python
import pandas as pd
from time import time

# Sample data with different characteristics
X_sample = pd.DataFrame({
    'data': [
        'a' * 100,           # Highly repetitive
        'abcdefghij' * 10,   # Pattern-based
        'random text here',   # Low redundancy
    ]
})
y_sample = pd.Series(['A', 'B', 'C'])

# Test each compressor
for comp_name in ['gzip', 'bz2', 'lzma']:
    start = time()
    knn = CompressionKNN(k=1, data_column='data', compressor=comp_name)
    knn.fit(X_sample, y_sample)
    elapsed = time() - start
    print(f"{comp_name.upper()}: {elapsed:.4f}s")
```

## <a name="custom-compressors"></a>Custom Compression Functions

You can provide your own compression function for specialized needs:

### Custom Compressor with zlib

```python
import zlib

def custom_compressor(data: bytes) -> bytes:
    """Custom compressor using maximum compression."""
    return zlib.compress(data, level=9)

knn = CompressionKNN(
    k=2,
    data_column='text',
    compressor=custom_compressor
)
```

### Requirements for Custom Compressors

Your function must:
1. Accept `bytes` as input
2. Return `bytes` as output
3. Be deterministic (same input → same output)

```python
from typing import Callable

def my_compressor(data: bytes) -> bytes:
    # Your compression logic here
    # Must return bytes
    return compressed_data
```

### Example: Combining Compressors

```python
import gzip
import bz2

def hybrid_compressor(data: bytes) -> bytes:
    """Try both and use the smaller result."""
    gzip_result = gzip.compress(data)
    bz2_result = bz2.compress(data)
    return gzip_result if len(gzip_result) < len(bz2_result) else bz2_result

knn = CompressionKNN(k=2, data_column='text', compressor=hybrid_compressor)
```

## <a name="data-types"></a>Working with Different Data Types

### Text Data (Default)

```python
X = pd.DataFrame({'text': ['Hello', 'World']})
knn = CompressionKNN(k=1, data_column='text', encoding='utf-8')
```

### Binary Data

If your data is already bytes, it works directly:

```python
X = pd.DataFrame({'binary': [b'data1', b'data2']})
knn = CompressionKNN(k=1, data_column='binary')
```

### Unicode and Special Characters

Handle non-ASCII text properly:

```python
X = pd.DataFrame({
    'text': [
        'Hello 世界',
        'Привет мир',
        'مرحبا بالعالم'
    ]
})

# Ensure UTF-8 encoding (default)
knn = CompressionKNN(k=1, data_column='text', encoding='utf-8')
```

### DNA Sequences

Perfect use case for compression-based classification:

```python
X_dna = pd.DataFrame({
    'sequence': [
        'ATCGATCGATCG' * 5,      # Bacteria
        'ATCGATCGATCG' * 5 + 'AT',
        'GCTAGCTAGCTA' * 5,
        'ATGCATGCATGC' * 3 + 'GGGGCCCC' * 2,  # Mammal
        'TAGCTAGCTAGC' * 3 + 'AAAATTTT' * 2,
    ]
})

y_dna = pd.Series(['Bacteria', 'Bacteria', 'Bacteria', 'Mammal', 'Mammal'])

# LZMA works well for pattern-rich data
knn_dna = CompressionKNN(k=3, data_column='sequence', compressor='lzma')
knn_dna.fit(X_dna, y_dna)
```

### Time Series Data

Convert time series to strings:

```python
import numpy as np

# Generate time series
ts1 = np.sin(np.linspace(0, 10, 100))
ts2 = np.cos(np.linspace(0, 10, 100))

# Convert to strings (preserving patterns)
X = pd.DataFrame({
    'series': [
        ''.join(map(str, ts1)),
        ''.join(map(str, ts2))
    ]
})
```

## <a name="performance"></a>Performance Tuning

### Choosing k

```python
# Smaller k = more sensitive to noise
knn_k1 = CompressionKNN(k=1, data_column='text')

# Larger k = smoother decision boundary
knn_k5 = CompressionKNN(k=5, data_column='text')

# Odd k preferred for binary classification (avoids ties)
knn_k3 = CompressionKNN(k=3, data_column='text')
```

### Speed Optimization

```python
# Use gzip for speed
knn_fast = CompressionKNN(k=1, data_column='text', compressor='gzip')

# Sample large datasets
if len(X_train) > 1000:
    X_train_sample = X_train.sample(n=1000, random_state=42)
    knn.fit(X_train_sample, y_train.iloc[X_train_sample.index])
```

### Memory Optimization

```python
# Use Polars for better memory efficiency
import polars as pl

X_train_pl = pl.DataFrame(X_train)
y_train_pl = pl.Series(y_train)

knn = CompressionKNN(k=2, data_column='text')
knn.fit(X_train_pl, y_train_pl)
```

## <a name="applications"></a>Real-World Applications

### Spam Detection

```python
X_spam = pd.DataFrame({
    'email': [
        'Congratulations! You won $1000000!!!',
        'Meeting at 3pm tomorrow',
        'BUY NOW!!! Limited offer!!!',
        'Can you send the report?',
        'CLICK HERE FOR FREE MONEY',
        'Lunch on Friday?'
    ]
})

y_spam = pd.Series(['spam', 'ham', 'spam', 'ham', 'spam', 'ham'])

clf_spam = CompressionKNN(k=3, data_column='email')
clf_spam.fit(X_spam, y_spam)
```

### Programming Language Detection

```python
X_code = pd.DataFrame({
    'code': [
        'def hello(): print("Hi")',
        'function hello() { console.log("Hi"); }',
        'void hello() { printf("Hi"); }',
        'print("Hello")',
        'console.log("Hello");',
    ]
})

y_code = pd.Series(['Python', 'JavaScript', 'C', 'Python', 'JavaScript'])

clf_lang = CompressionKNN(k=2, data_column='code')
clf_lang.fit(X_code, y_code)
```

### Authorship Attribution

```python
# Identify writing style
X_author = pd.DataFrame({
    'text': [
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael.",
        "It is a truth universally acknowledged...",
        "In a hole in the ground there lived a hobbit.",
        # ... more samples
    ]
})

y_author = pd.Series(['Dickens', 'Melville', 'Austen', 'Tolkien'])

clf_author = CompressionKNN(k=3, data_column='text', compressor='lzma')
clf_author.fit(X_author, y_author)
```

## Best Practices

### ✅ Do:
- Use odd k for binary classification
- Try different compressors for your data type
- Normalize/clean your text before classification
- Use LZMA for pattern-heavy data
- Keep datasets relatively small (< 10,000 samples)

### ❌ Don't:
- Use compression KNN for numerical feature vectors (use standard KNN)
- Expect real-time performance on large datasets
- Mix different data types in the same column
- Forget to handle encoding issues with non-ASCII text

## Debugging Tips

### Check Compression Sizes

```python
import gzip

text1 = "Hello world"
text2 = "Hello there"

print(f"Text1 compressed: {len(gzip.compress(text1.encode()))}")
print(f"Text2 compressed: {len(gzip.compress(text2.encode()))}")
print(f"Concat compressed: {len(gzip.compress((text1 + text2).encode()))}")
```

### Visualize Distances

```python
from explodingham.utils.distance_metrics.ncd import NormalizedCompressionDistance
import gzip

ncd = NormalizedCompressionDistance(gzip.compress)

texts = ['Hello', 'Hi', 'Hola', 'Bonjour']
for i, t1 in enumerate(texts):
    for t2 in texts[i+1:]:
        dist = ncd.ncd(t1.encode(), t2.encode())
        print(f"NCD('{t1}', '{t2}') = {dist:.3f}")
```

## Next Steps

- **[DNA Sequence Classification Example](../examples/dna_sequences.md)** - Bioinformatics application
- **[Text Classification Example](../examples/text_classification.md)** - More text applications
- **[API Reference](../api/models/compression_learning.md)** - Complete parameter documentation

---
