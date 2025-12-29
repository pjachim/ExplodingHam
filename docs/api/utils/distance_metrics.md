# utils.distance_metrics

Distance and similarity metrics for machine learning.

## NormalizedCompressionDistance

```python
from explodingham.utils.distance_metrics.ncd import NormalizedCompressionDistance
```

Compute the Normalized Compression Distance (NCD) between two strings.

### Class Signature

```python
NormalizedCompressionDistance(
    compressor: Callable[[str | bytes], str | bytes]
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `compressor` | Callable[[str \| bytes], str \| bytes] | Compression function that takes string or bytes and returns compressed data |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `compressor` | Callable | The compression function used for computing distances |

### Methods

#### compress_size

```python
compress_size(data: str | bytes) -> int
```

Get the compressed size of the given data.

**Parameters:**
- `data` (str or bytes): Input data to compress

**Returns:**
- `size` (int): Size in bytes of compressed data

**Example:**
```python
import zlib
ncd = NormalizedCompressionDistance(zlib.compress)
size = ncd.compress_size(b"hello world")
# Returns compressed size as int
```

#### ncd

```python
ncd(x: str | bytes, y: str | bytes) -> float
```

Calculate the Normalized Compression Distance between two objects.

**Parameters:**
- `x` (str or bytes): First object to compare
- `y` (str or bytes): Second object to compare

**Returns:**
- `distance` (float): NCD value, approximately between 0 (identical) and 1 (completely different)

**Example:**
```python
import zlib
ncd_metric = NormalizedCompressionDistance(zlib.compress)

dist = ncd_metric.ncd(b"hello world", b"hello there")
print(f"Distance: {dist:.4f}")
# Distance: 0.4545
```

### Theory

NCD is a similarity metric based on Kolmogorov complexity theory. It measures information distance by comparing compressed sizes.

**Formula:**

$$
NCD(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}
$$

where:
- $C(x)$ = compressed size of x
- $xy$ = concatenation of x and y

**Properties:**
- **Range:** Approximately [0, 1+] (exact upper bound depends on compressor)
- **Symmetry:** NCD(x, y) ≈ NCD(y, x) (approximately symmetric)
- **Triangle inequality:** Approximately satisfied
- **Universal:** Works for any data type

### Examples

#### Basic Usage with zlib

```python
import zlib
from explodingham.utils.distance_metrics.ncd import NormalizedCompressionDistance

ncd = NormalizedCompressionDistance(zlib.compress)

# Compare similar strings
dist1 = ncd.ncd(b"hello world", b"hello there")
print(f"Similar: {dist1:.4f}")  # ~0.45

# Compare identical strings
dist2 = ncd.ncd(b"test", b"test")
print(f"Identical: {dist2:.4f}")  # 0.0

# Compare different strings
dist3 = ncd.ncd(b"apple", b"zebra")
print(f"Different: {dist3:.4f}")  # ~1.0
```

#### Using Different Compressors

```python
import gzip
import bz2
import lzma

# Test with different compressors
doc1 = b"The quick brown fox jumps over the lazy dog"
doc2 = b"The quick brown fox jumps over the sleepy cat"

for comp_name, comp_func in [('gzip', gzip.compress), 
                               ('bz2', bz2.compress),
                               ('lzma', lzma.compress)]:
    ncd = NormalizedCompressionDistance(comp_func)
    dist = ncd.ncd(doc1, doc2)
    similarity = 1 - dist
    print(f"{comp_name:5}: {similarity:.2%} similar")
```

#### Text Similarity

```python
import bz2

ncd = NormalizedCompressionDistance(bz2.compress)

# Language detection via similarity
english = b"The cat sat on the mat"
spanish = b"El gato se sentó en la alfombra"
french = b"Le chat s'est assis sur le tapis"

# Compare Spanish to English vs French
dist_es_en = ncd.ncd(spanish, english)
dist_es_fr = ncd.ncd(spanish, french)

print(f"Spanish-English distance: {dist_es_en:.3f}")
print(f"Spanish-French distance: {dist_es_fr:.3f}")
# French and Spanish likely more similar due to Romance language roots
```

#### DNA Sequence Similarity

```python
import lzma

ncd = NormalizedCompressionDistance(lzma.compress)

# Compare DNA sequences
seq1 = b"ATCGATCGATCGATCG" * 5
seq2 = b"ATCGATCGATCGATCG" * 5 + b"AT"
seq3 = b"GGCCGGCCGGCCGGCC" * 5

dist_1_2 = ncd.ncd(seq1, seq2)
dist_1_3 = ncd.ncd(seq1, seq3)

print(f"Similar sequences: {dist_1_2:.3f}")  # Low distance
print(f"Different sequences: {dist_1_3:.3f}")  # Higher distance
```

#### Clustering with NCD

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Sample documents
docs = [
    b"Python programming language",
    b"Python is great for coding",
    b"Java programming language",
    b"JavaScript web development",
    b"Machine learning with Python"
]

# Compute pairwise NCD matrix
n = len(docs)
dist_matrix = np.zeros((n, n))

ncd_metric = NormalizedCompressionDistance(gzip.compress)

for i in range(n):
    for j in range(i+1, n):
        dist = ncd_metric.ncd(docs[i], docs[j])
        dist_matrix[i][j] = dist
        dist_matrix[j][i] = dist

# Hierarchical clustering
# Convert to condensed distance matrix for scipy
from scipy.spatial.distance import squareform
condensed = squareform(dist_matrix)
linkage_matrix = linkage(condensed, method='average')

# Plot dendrogram
# dendrogram(linkage_matrix, labels=[d.decode()[:20] for d in docs])
# plt.show()
```

### Use Cases

**Good for:**
- ✅ Text similarity without tokenization
- ✅ Sequence comparison (DNA, proteins)
- ✅ File similarity detection
- ✅ Language-agnostic similarity
- ✅ Clustering based on structure

**Limitations:**
- ❌ Slow (compression is expensive)
- ❌ Not suitable for real-time applications
- ❌ Approximate metric (not true mathematical metric)
- ❌ Depends on compressor choice

### Compressor Choice

| Compressor | Speed | Best For |
|------------|-------|----------|
| gzip | ⚡⚡⚡ | General text, fast computation |
| bz2 | ⚡⚡ | Repetitive patterns |
| lzma | ⚡ | Complex patterns, best accuracy |

### Mathematical Properties

**Normalization:**  
NCD is normalized to account for different object sizes, making it suitable for comparing objects of varying lengths.

**Symmetry:**  
While theoretically symmetric, practical compression can lead to small asymmetries: NCD(x,y) ≈ NCD(y,x)

### See Also

- [CompressionKNN](../models/compression_learning.md#compressionknn) - Uses NCD for classification
- [Tutorial: Compression KNN](../../tutorials/compression_knn.md) - Detailed guide
- [Examples: Custom Compressors](../../examples/custom_compressors.md) - Building custom compressors

---

NCD provides a parameter-free, universal similarity metric based on solid information-theoretic foundations. Use it when traditional distance metrics (Euclidean, cosine) aren't appropriate for your sequential or structural data.
