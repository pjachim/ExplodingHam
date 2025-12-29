# Custom Compressors

While ExplodingHam provides options for compressors offered in base python (gzip, bz2, lzma), you can create custom compression functions tailored to your specific data and use case.

## Requirements for Custom Compressors

Your compression function must:

1. **Accept `bytes` as input**
2. **Return `bytes` as output**

```python
from typing import Callable

def my_compressor(data: bytes) -> bytes:
    """Template for custom compressor."""
    # Your compression logic here
    compressed_data = ...  # Must return bytes
    return compressed_data
```

## Basic Custom Compressor

### Using zlib with Custom Settings

```python
import zlib
from explodingham.models.compression_learning.knn import CompressionKNN
import pandas as pd

def max_compression(data: bytes) -> bytes:
    """Use maximum compression level."""
    return zlib.compress(data, level=9)

def fast_compression(data: bytes) -> bytes:
    """Use fast compression."""
    return zlib.compress(data, level=1)

# Compare compression levels
X_train = pd.DataFrame({
    'text': ['Hello world', 'Bonjour monde', 'Hola mundo']
})
y_train = pd.Series(['English', 'French', 'Spanish'])

# Maximum compression (slower, better accuracy)
clf_max = CompressionKNN(k=1, data_column='text', compressor=max_compression)
clf_max.fit(X_train, y_train)

# Fast compression (faster, potentially lower accuracy)
clf_fast = CompressionKNN(k=1, data_column='text', compressor=fast_compression)
clf_fast.fit(X_train, y_train)
```

## Domain-Specific Compressors

### DNA-Specific Compression

```python
def dna_compressor(data: bytes) -> bytes:
    """
    Simple 2-bit encoding for DNA sequences.
    A=00, C=01, G=10, T=11
    """
    # Decode bytes to string
    seq = data.decode('utf-8')
    
    # Map nucleotides to 2-bit codes
    mapping = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
    
    # Convert to bit string
    bit_string = ''.join(mapping.get(base, '00') for base in seq.upper())
    
    # Convert bit string to bytes
    # Pad to make length divisible by 8
    while len(bit_string) % 8 != 0:
        bit_string += '0'
    
    # Convert to bytes
    compressed = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))
    
    return compressed

# Use DNA compressor
dna_clf = CompressionKNN(
    k=2,
    data_column='sequence',
    compressor=dna_compressor
)

X_dna = pd.DataFrame({
    'sequence': ['ATCGATCG', 'GCTAGCTA', 'ATATATATA']
})
y_dna = pd.Series(['Type1', 'Type2', 'Type1'])

dna_clf.fit(X_dna, y_dna)
```

### Text-Specific Run-Length Encoding

```python
def run_length_encode(data: bytes) -> bytes:
    """
    Simple run-length encoding for repetitive text.
    Example: 'AAAA' -> '4A'
    """
    text = data.decode('utf-8')
    
    if not text:
        return b''
    
    encoded = []
    current_char = text[0]
    count = 1
    
    for char in text[1:]:
        if char == current_char:
            count += 1
        else:
            encoded.append(f"{count}{current_char}")
            current_char = char
            count = 1
    
    # Add last run
    encoded.append(f"{count}{current_char}")
    
    return ''.join(encoded).encode('utf-8')

# Use RLE compressor
rle_clf = CompressionKNN(
    k=1,
    data_column='text',
    compressor=run_length_encode
)
```

## Hybrid Compressors

### Best-of-Multiple Compressors

```python
import gzip
import bz2
import lzma

def hybrid_compressor(data: bytes) -> bytes:
    """
    Try multiple compressors and use the best result.
    Returns the smallest compressed output.
    """
    results = {
        'gzip': gzip.compress(data),
        'bz2': bz2.compress(data),
        'lzma': lzma.compress(data)
    }
    
    # Return the smallest
    return min(results.values(), key=len)

# Use hybrid compressor
hybrid_clf = CompressionKNN(
    k=2,
    data_column='text',
    compressor=hybrid_compressor
)

X = pd.DataFrame({'text': ['test data', 'more test data']})
y = pd.Series(['A', 'B'])

hybrid_clf.fit(X, y)
```

### Cascaded Compression

```python
def cascaded_compressor(data: bytes) -> bytes:
    """
    Apply multiple compression algorithms in sequence.
    Sometimes this can achieve better compression.
    """
    # First pass: gzip
    compressed = gzip.compress(data)
    
    # Second pass: bz2 on the gzip output
    compressed = bz2.compress(compressed)
    
    return compressed

cascaded_clf = CompressionKNN(
    k=2,
    data_column='text',
    compressor=cascaded_compressor
)
```

## Preprocessing Compressors

### Normalization + Compression

```python
import gzip

def normalized_compressor(data: bytes) -> bytes:
    """
    Normalize text before compressing.
    Useful for text where case and whitespace don't matter.
    """
    # Decode, normalize, encode
    text = data.decode('utf-8')
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Compress normalized text
    return gzip.compress(text.encode('utf-8'))

norm_clf = CompressionKNN(
    k=2,
    data_column='text',
    compressor=normalized_compressor
)
```

### Tokenization + Compression

```python
import gzip

def token_compressor(data: bytes) -> bytes:
    """
    Tokenize before compressing for better pattern detection.
    """
    text = data.decode('utf-8')
    
    # Simple word tokenization
    tokens = text.lower().split()
    
    # Join with consistent separator
    tokenized = ' '.join(sorted(tokens))  # Sort for consistency
    
    return gzip.compress(tokenized.encode('utf-8'))

token_clf = CompressionKNN(
    k=2,
    data_column='text',
    compressor=token_compressor
)
```

## Specialized Compressors

### Dictionary-Based Compression

```python
class DictionaryCompressor:
    """
    Custom dictionary-based compressor for domain-specific text.
    """
    def __init__(self, dictionary):
        """
        dictionary: dict mapping common phrases to short codes
        """
        self.dictionary = dictionary
        self.reverse_dict = {v: k for k, v in dictionary.items()}
    
    def compress(self, data: bytes) -> bytes:
        """Compress using dictionary substitution."""
        text = data.decode('utf-8')
        
        # Replace dictionary entries
        for phrase, code in self.dictionary.items():
            text = text.replace(phrase, code)
        
        # Then apply standard compression
        return gzip.compress(text.encode('utf-8'))

# Create domain-specific dictionary
medical_dict = {
    'patient': 'PT',
    'diagnosis': 'DX',
    'treatment': 'TX',
    'symptoms': 'SX'
}

dict_compressor = DictionaryCompressor(medical_dict)

# Use with classifier
medical_clf = CompressionKNN(
    k=2,
    data_column='note',
    compressor=dict_compressor.compress
)
```

### Context-Aware Compression

```python
import gzip

class ContextCompressor:
    """
    Compressor that uses context/metadata for better compression.
    """
    def __init__(self, context=''):
        self.context = context
    
    def compress(self, data: bytes) -> bytes:
        """Prepend context before compressing."""
        text = data.decode('utf-8')
        
        # Add context (helps compression find patterns)
        combined = self.context + ' ' + text
        
        return gzip.compress(combined.encode('utf-8'))

# Create context-aware compressor for specific domain
code_compressor = ContextCompressor(context='def class import function')

code_clf = CompressionKNN(
    k=2,
    data_column='code',
    compressor=code_compressor.compress
)
```

## Optimized Compressors

### Memory-Efficient Compressor

```python
import gzip
from io import BytesIO

def memory_efficient_compressor(data: bytes) -> bytes:
    """
    Use streaming compression for large data.
    """
    # Use BytesIO for in-memory streaming
    buffer = BytesIO()
    
    with gzip.GzipFile(fileobj=buffer, mode='wb') as gz:
        gz.write(data)
    
    return buffer.getvalue()

efficient_clf = CompressionKNN(
    k=2,
    data_column='text',
    compressor=memory_efficient_compressor
)
```

### Cached Compressor

```python
from functools import lru_cache
import gzip

# Cache compression results (trade memory for speed)
@lru_cache(maxsize=10000)
def cached_compressor(data: bytes) -> bytes:
    """Cache compression results to avoid recomputing."""
    return gzip.compress(data)

# Note: data must be hashable (bytes are hashable)
cached_clf = CompressionKNN(
    k=2,
    data_column='text',
    compressor=cached_compressor
)
```

## Testing Custom Compressors

### Compression Ratio Test

```python
def test_compressor(compressor, data_samples):
    """Test compression ratio on sample data."""
    for data in data_samples:
        original_size = len(data.encode('utf-8'))
        compressed_size = len(compressor(data.encode('utf-8')))
        ratio = compressed_size / original_size
        print(f"Ratio: {ratio:.3f} | {data[:30]}...")

# Test samples
samples = [
    "Hello world hello world hello world",
    "ATCGATCGATCGATCGATCG",
    "Random text without patterns"
]

test_compressor(max_compression, samples)
```

### Compression Speed Test

```python
import time

def benchmark_compressor(compressor, data, iterations=100):
    """Benchmark compressor speed."""
    data_bytes = data.encode('utf-8')
    
    start = time.time()
    for _ in range(iterations):
        _ = compressor(data_bytes)
    elapsed = time.time() - start
    
    print(f"Time for {iterations} compressions: {elapsed:.3f}s")
    print(f"Average: {elapsed/iterations*1000:.2f}ms")

benchmark_compressor(max_compression, "Test data " * 100)
```

## Best Practices

### ✅ Do:

1. **Test compression ratio**: Ensure your compressor actually compresses
2. **Benchmark speed**: Know the performance tradeoff
3. **Handle edge cases**: Empty strings, special characters
4. **Be deterministic**: Same input = same output always
5. **Document behavior**: Explain what makes your compressor special

### ❌ Don't:

1. **Return lossy compression**: Must be able to decompress exactly
2. **Make non-deterministic**: No randomness or timestamps
3. **Ignore encoding**: Handle bytes/string conversion properly
4. **Over-optimize prematurely**: Start simple, optimize if needed
5. **Forget error handling**: Handle invalid input gracefully

## Example: Complete Custom Compressor

```python
import gzip
import re

class SmartTextCompressor:
    """
    Intelligent text compressor with preprocessing.
    """
    def __init__(
        self,
        lowercase=True,
        remove_punctuation=True,
        normalize_whitespace=True,
        compression_level=6
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
        self.compression_level = compression_level
    
    def preprocess(self, text: str) -> str:
        """Preprocess text before compression."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        if self.normalize_whitespace:
            text = ' '.join(text.split())
        
        return text
    
    def compress(self, data: bytes) -> bytes:
        """Compress with preprocessing."""
        # Decode and preprocess
        text = data.decode('utf-8', errors='ignore')
        text = self.preprocess(text)
        
        # Compress
        return gzip.compress(
            text.encode('utf-8'),
            compresslevel=self.compression_level
        )

# Use smart compressor
smart_compressor = SmartTextCompressor(
    lowercase=True,
    remove_punctuation=True,
    normalize_whitespace=True,
    compression_level=9
)

smart_clf = CompressionKNN(
    k=3,
    data_column='text',
    compressor=smart_compressor.compress
)

X = pd.DataFrame({
    'text': [
        'Hello, World!!!',
        'hello world',
        'HELLO WORLD'
    ]
})
y = pd.Series(['A', 'A', 'A'])

smart_clf.fit(X, y)
```

## When to Build Custom Compressors

**Consider a custom compressor when:**

- ✅ You have domain-specific knowledge about data structure
- ✅ Standard compressors don't capture important patterns
- ✅ You need specific preprocessing (normalization, tokenization)
- ✅ Performance is critical and you can optimize
- ✅ You want to experiment with novel compression approaches

**Stick with built-ins when:**

- ❌ You're just getting started
- ❌ Standard compressors work well enough
- ❌ You don't have strong hypotheses about data structure
- ❌ Development time is limited

## Next Steps

- **[Compression KNN Tutorial](../tutorials/compression_knn.md)** - Deep dive into compression-based classification
- **[API Reference](../api/models/compression_learning.md)** - Complete parameter documentation
- **[Text Classification Examples](text_classification.md)** - Apply custom compressors

---

Custom compressors let you inject domain knowledge into the classification process. Start with built-ins, then optimize with custom compressors when needed!
