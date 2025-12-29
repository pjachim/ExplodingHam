# Getting Started with ExplodingHam

This guide will walk you through your first steps with ExplodingHam, demonstrating the core concepts and showing you how to use our models.

## Basic Concepts

ExplodingHam models follow the familiar scikit-learn API:

1. **Import** the model
2. **Initialize** with parameters
3. **Fit** on training data
4. **Predict** on new data

All models are compatible with both Pandas and Polars DataFrames thanks to the Narwhals library.

## Your First Model: Text Classification

Let's build a simple language classifier using compression-based KNN:

### Step 1: Import and Prepare Data

```python
from explodingham.models.compression_learning.knn import CompressionKNN
import pandas as pd

# Create training data
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

# Create test data
X_test = pd.DataFrame({
    'text': [
        'Hi there, what is your name?',
        'Hola amigo, qué tal?',
        'Bonjour, ça va?'
    ]
})
```

### Step 2: Create and Train the Model

```python
# Initialize the classifier
clf = CompressionKNN(
    k=2,              # Use 2 nearest neighbors
    data_column='text',  # Column containing text to analyze
    compressor='gzip'    # Compression algorithm (default)
)

# Train the model
clf.fit(X_train, y_train)
print("✓ Model trained!")
```

### Step 3: Make Predictions

```python
# Predict on test data
predictions = clf.predict(X_test)

print("\nPredictions:")
for text, prediction in zip(X_test['text'], predictions):
    print(f"  '{text}' -> {prediction}")
```

**Expected Output:**
```
✓ Model trained!

Predictions:
  'Hi there, what is your name?' -> English
  'Hola amigo, qué tal?' -> Spanish
  'Bonjour, ça va?' -> French
```

## How It Works

### Compression-Based Classification

CompressionKNN uses a brilliant insight from information theory: **similar objects compress well together**.

When you concatenate two similar strings and compress them, the result is smaller than compressing each separately and adding the sizes. This is because compression algorithms exploit patterns and redundancy.

The **Normalized Compression Distance (NCD)** quantifies this:

```
NCD(x, y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))
```

where:
- `C(x)` = compressed size of x
- `xy` = concatenation of x and y

For classification:
1. Compress each training example
2. For a test example, compress it concatenated with each training example
3. Find the k nearest neighbors (smallest NCD values)
4. Predict the majority class among those neighbors

## Using Different Compressors

ExplodingHam supports three built-in compressors:

```python
# Fast, good general-purpose
clf_gzip = CompressionKNN(k=2, data_column='text', compressor='gzip')

# Better compression ratio, slower
clf_bz2 = CompressionKNN(k=2, data_column='text', compressor='bz2')

# Best compression ratio, slowest
clf_lzma = CompressionKNN(k=2, data_column='text', compressor='lzma')
```

You can also provide your own compression function:

```python
import zlib

def custom_compressor(data: bytes) -> bytes:
    return zlib.compress(data, level=9)

clf_custom = CompressionKNN(
    k=2,
    data_column='text',
    compressor=custom_compressor
)
```

## Working with Polars

ExplodingHam works seamlessly with Polars:

```python
import polars as pl

# Create Polars DataFrames
X_train_pl = pl.DataFrame({
    'text': ['Python is great', 'JavaScript is fun']
})
y_train_pl = pl.Series('label', ['Python', 'JavaScript'])

X_test_pl = pl.DataFrame({
    'text': ['Python rocks']
})

# Same API!
clf = CompressionKNN(k=1, data_column='text')
clf.fit(X_train_pl, y_train_pl)
predictions = clf.predict(X_test_pl)
```

## Quick Baseline with Regex

Sometimes you need a quick baseline before diving into ML. The `BinaryRegexClassifier` lets you create a scikit-learn-compatible classifier from a regex pattern:

```python
from explodingham.models.baseline_models.regex import BinaryRegexClassifier

# Create a classifier to detect URLs
url_clf = BinaryRegexClassifier(
    pattern=r'https?://',
    match_type='partial',
    match_prediction='URL',
    no_match_prediction='Not URL'
)

# Use it like any sklearn classifier
data = pd.Series(['https://example.com', 'just text', 'http://site.org'])
predictions = url_clf.predict(data)['prediction']
# ['URL', 'Not URL', 'URL']
```

## Key Parameters Explained

### CompressionKNN

- **`k`**: Number of neighbors to consider. Larger k = smoother boundaries but potentially less accurate
- **`data_column`**: Which column contains the data to compress
- **`compressor`**: Which compression algorithm to use (`'gzip'`, `'bz2'`, `'lzma'`, or a custom function)
- **`encoding`**: How to encode strings to bytes (default: `'utf-8'`)

### BinaryRegexClassifier

- **`pattern`**: The regex pattern to match
- **`match_type`**: `'partial'` (pattern anywhere) or `'full'` (entire string matches)
- **`match_prediction`**: Value to return when pattern matches
- **`no_match_prediction`**: Value to return when pattern doesn't match
- **`ignore_case`**: Whether to ignore case in matching

## Common Patterns

### Loading Data

```python
import pandas as pd

# From CSV
df = pd.read_csv('data.csv')
X = df[['text_column']]
y = df['label']

# From lists
X = pd.DataFrame({'text': ['sample 1', 'sample 2']})
y = pd.Series(['A', 'B'])
```

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Detailed report
print(classification_report(y_test, y_pred))
```

## When to Use Each Model

### Use CompressionKNN when:
- ✅ You have text, DNA sequences, or sequential data
- ✅ You want to avoid feature engineering
- ✅ Dataset is small-to-medium (compression is slow)
- ✅ You need a universal similarity metric

### Use BinaryRegexClassifier when:
- ✅ You need a quick baseline
- ✅ You have clear patterns you can express in regex
- ✅ You want to test your ML against a simple rule
- ✅ You're building a pipeline and want a non-ML fallback

## Next Steps

Now that you understand the basics:

- **[Compression KNN Tutorial](tutorials/compression_knn.md)** - Deep dive into compression-based classification
- **[Text Classification Examples](examples/text_classification.md)** - Real-world applications
- **[API Reference](api/models.md)** - Complete parameter documentation

## Common Issues

### Performance

CompressionKNN can be slow on large datasets because it compresses every pair of examples. For better performance:

- Use `compressor='gzip'` (fastest)
- Keep k small (e.g., k=1 or k=3)
- Consider sampling your training data

### Memory

If you run into memory issues:

- Process data in batches
- Use Polars instead of Pandas (more memory efficient)
- Reduce the size of your training set

### Accuracy

If accuracy is low:

- Try different compressors (`'lzma'` often works better for pattern-heavy data)
- Experiment with different k values
- Ensure your data column doesn't have extra whitespace or formatting issues

---

**Ready for more?** Check out the [tutorials](tutorials/compression_knn.md) for in-depth guides!
