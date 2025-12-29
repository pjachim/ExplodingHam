# models.compression_learning

Compression-based machine learning algorithms that use information theory for classification.

## CompressionKNN

```python
from explodingham.models.compression_learning.knn import CompressionKNN
```

K-Nearest Neighbors classifier using Normalized Compression Distance (NCD) as the similarity metric.

### Class Signature

```python
CompressionKNN(
    k: int,
    data_column: str | None = None,
    encoding: str = 'utf-8',
    compressor: str | Callable = 'gzip',
    encoded: bool = False,
    target_column: str = 'target'
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | **required** | Number of nearest neighbors to use for classification. Must be positive. Odd values preferred for binary classification to avoid ties. |
| `data_column` | str or None | None | Name of the column containing data to compress. If None, inferred from first non-target column. |
| `encoding` | str | 'utf-8' | Character encoding for converting strings to bytes. Options: 'utf-8', 'ascii', 'latin-1', etc. |
| `compressor` | str or Callable | 'gzip' | Compression method. String options: 'gzip' (fast, general-purpose), 'bz2' (better compression, slower), 'lzma' (best compression, slowest). Or provide a custom function with signature `f(bytes) -> bytes`. |
| `encoded` | bool | False | If True, assumes input data is already bytes and skips encoding. |
| `target_column` | str | 'target' | Name of the target label column in training data. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_data` | nw.DataFrame | Training data with precomputed compressed lengths |
| `compressor` | Callable | Compression function used for NCD computation |
| `encoding` | str | Character encoding for string-to-bytes conversion |
| `k` | int | Number of nearest neighbors |
| `target_column` | str | Name of target variable column |

### Methods

#### fit

```python
fit(X_train: DataFrameOrSeries, y_train: DataFrameOrSeries) -> CompressionKNN
```

Fit the classifier on training data.

**Parameters:**
- `X_train` (DataFrame or Series): Training samples. Can be DataFrame with one or more columns or Series.
- `y_train` (DataFrame or Series): Target labels corresponding to X_train. Must have same number of rows.

**Returns:**
- `self` (CompressionKNN): Fitted classifier instance (enables method chaining).

**Example:**
```python
X_train = pd.DataFrame({'text': ['cat', 'dog', 'mouse']})
y_train = pd.Series(['mammal', 'mammal', 'mammal'])

knn = CompressionKNN(k=2, data_column='text')
knn.fit(X_train, y_train)
```

#### predict

```python
predict(X_test: DataFrameOrSeries) -> Series
```

Predict class labels for test data.

**Parameters:**
- `X_test` (DataFrame or Series): Test samples to classify. Must have same structure as X_train.

**Returns:**
- `predictions` (Series): Predicted class labels for each test sample.

**Example:**
```python
X_test = pd.DataFrame({'text': ['puppy', 'kitten']})
predictions = knn.predict(X_test)
# Series(['mammal', 'mammal'])
```

### Examples

#### Basic Text Classification

```python
from explodingham.models.compression_learning.knn import CompressionKNN
import pandas as pd

# Training data
X_train = pd.DataFrame({
    'text': [
        'Hello world',
        'Good morning',
        'Hola mundo',
        'Buenos días'
    ]
})
y_train = pd.Series(['English', 'English', 'Spanish', 'Spanish'])

# Create and train
knn = CompressionKNN(k=1, data_column='text')
knn.fit(X_train, y_train)

# Predict
X_test = pd.DataFrame({'text': ['Hi there', 'Hola amigo']})
predictions = knn.predict(X_test)
# ['English', 'Spanish']
```

#### Custom Compressor

```python
import zlib

def custom_compressor(data: bytes) -> bytes:
    return zlib.compress(data, level=9)

knn = CompressionKNN(
    k=2,
    data_column='text',
    compressor=custom_compressor
)
knn.fit(X_train, y_train)
```

#### DNA Sequence Classification

```python
X_dna = pd.DataFrame({
    'seq': [
        'ATCGATCG' * 10,
        'ATCGATCG' * 10 + 'AT',
        'GCTAGCTA' * 8 + 'GGGG',
        'TAGCTACG' * 8 + 'CCCC'
    ]
})
y_dna = pd.Series(['Bacteria', 'Bacteria', 'Mammal', 'Mammal'])

knn = CompressionKNN(k=1, data_column='seq', compressor='lzma')
knn.fit(X_dna, y_dna)
```

#### With Polars

```python
import polars as pl

X_train_pl = pl.DataFrame({
    'text': ['Python is great', 'JavaScript is fun']
})
y_train_pl = pl.Series('label', ['Python', 'JavaScript'])

knn = CompressionKNN(k=1, data_column='text')
knn.fit(X_train_pl, y_train_pl)
```

### Theory

CompressionKNN uses **Normalized Compression Distance (NCD)** to measure similarity:

$$
NCD(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}
$$

where:
- $C(x)$ = compressed size of x
- $xy$ = concatenation of x and y

**Key Insight:** Similar objects compress well together because compression algorithms exploit patterns and redundancy.

### Performance Characteristics

**Time Complexity:**
- Training: O(n) where n = number of training samples
- Prediction: O(n * m * c) where m = number of test samples, c = compression time

**Space Complexity:** O(n) - stores all training data in memory

### Advantages

✅ **No feature engineering required** - works directly on raw text/sequences  
✅ **Parameter-free distance** - only need to choose k  
✅ **DataFrame agnostic** - works with Pandas and Polars  

### Limitations

❌ **Computationally expensive** - compression is slow  
❌ **Not suitable for large datasets** - quadratic complexity  
❌ **Memory intensive** - stores all training data  
❌ **Word overlap bias** - shared substrings may create unfair advantages  

### When to Use

**Use CompressionKNN when:**
- You have a single text/bytes feature that you want to perform your analysis on.
- Dataset is small-to-medium (<10,000 samples), can also handle extremely small training sets (e.g. 5 samples per class).
- You don't need much interpretability.

**Don't use when:**
- Dataset is very large (>100,000 samples)
- Real-time predictions are needed
- You have numerical feature vectors (use standard KNN), or multiple features.
- Word-token features would work better

### References

Based on the paper:

Zhiying Jiang, Matthew Yang, Mikhail Tsirlin, Raphael Tang, Yiqin Dai, and Jimmy Lin. 2023. "'Low-Resource' Text Classification: A Parameter-Free Classification Method with Compressors." In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 6810-6828, Toronto, Canada. Association for Computational Linguistics. [https://aclanthology.org/2023.findings-acl.426](https://aclanthology.org/2023.findings-acl.426)

### See Also

- [NormalizedCompressionDistance](../utils/distance_metrics.md#normalizedcompressiondistance) - Standalone NCD metric
- [BaseKNNModel](#baseknnmodel) - Base class for KNN models
- [Tutorial: Compression KNN](../../tutorials/compression_knn.md) - In-depth guide
- [Examples: Text Classification](../../examples/text_classification.md) - Real-world applications

---