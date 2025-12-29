# models.baseline_models

Non-ML baseline classifiers for quick comparisons and benchmarking.

## BinaryRegexClassifier

```python
from explodingham.models.baseline_models.regex import BinaryRegexClassifier
```

DataFrame-aware regex classifier that matches patterns using configurable match types.

### Class Signature

```python
BinaryRegexClassifier(
    pattern: str = '',
    flags: list[re.RegexFlag] | None = None,
    ignore_case: bool = False,
    encoding: str = 'utf-8',
    column_name: str | None = None,
    match_prediction: Any = 1,
    no_match_prediction: Any = 0,
    match_type: str = 'full',
    prediction_name: str = 'prediction'
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | str | '' | Regular expression pattern to match |
| `flags` | list[re.RegexFlag] or None | None | Regex flags (IGNORECASE, MULTILINE, DOTALL, VERBOSE, UNICODE) |
| `ignore_case` | bool | False | Convenience parameter to add IGNORECASE flag |
| `encoding` | str | 'utf-8' | Character encoding for string operations |
| `column_name` | str or None | None | Column to match against. If None, input must be Series or single-column DataFrame |
| `match_prediction` | Any | 1 | Value returned when pattern matches |
| `no_match_prediction` | Any | 0 | Value returned when pattern doesn't match |
| `match_type` | str | 'full' | Match type: 'partial' (pattern anywhere) or 'full' (entire string matches) |
| `prediction_name` | str | 'prediction' | Name of output prediction column |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `pattern` | str | Compiled pattern with inline flags |
| `flags` | list[re.RegexFlag] | List of regex flags being used |
| `encoding` | str | Character encoding |
| `column_name` | str or None | Target column name |
| `match_prediction` | Any | Value for matches |
| `no_match_prediction` | Any | Value for non-matches |
| `match_type` | str | Type of matching |
| `prediction_name` | str | Output column name |

### Methods

#### fit

```python
fit(X: Any = None, y: Any = None) -> None
```

Fit method for sklearn API compatibility. No training required for regex classifiers.

**Parameters:**
- `X` (Any, optional): Input data (not used)
- `y` (Any, optional): Target labels (not used)

**Note:** This method does nothing - regex classifiers don't learn from data.

#### predict

```python
predict(X: DataFrame | Series) -> DataFrame
```

Apply regex pattern matching and return predictions.

**Parameters:**
- `X` (DataFrame or Series): Input data to classify. Can be pandas/polars DataFrame/Series

**Returns:**
- `predictions` (DataFrame): DataFrame with prediction column containing match/no-match values

**Example:**
```python
texts = pd.Series(['https://example.com', 'no url here'])
results = clf.predict(texts)
# DataFrame with 'prediction' column: [1, 0]
```

### Examples

#### URL Detection

```python
from explodingham.models.baseline_models.regex import BinaryRegexClassifier
import pandas as pd

# Detect URLs
url_clf = BinaryRegexClassifier(
    pattern=r'https?://',
    match_type='partial'
)

urls = pd.Series([
    'https://example.com',
    'http://site.org',
    'ftp://files.com',
    'example.com'
])

results = url_clf.predict(urls)
print(results['prediction'])
# [1, 1, 0, 0]
```

#### Email Validation

```python
# Full match for strict validation
email_clf = BinaryRegexClassifier(
    pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    match_type='partial',
    match_prediction='VALID',
    no_match_prediction='INVALID'
)

emails = pd.Series([
    'support@example.com',
    'not an email',
    'user@domain.co.uk'
])

validation = email_clf.predict(emails)
print(validation['prediction'])
# ['VALID', 'INVALID', 'VALID']
```

#### Case-Insensitive Search

```python
# Detect keyword regardless of case
keyword_clf = BinaryRegexClassifier(
    pattern=r'python',
    match_type='partial',
    ignore_case=True,
    match_prediction='TECH',
    no_match_prediction='OTHER'
)

texts = pd.Series([
    'I love Python',
    'PYTHON rocks',
    'Java is nice'
])

categories = keyword_clf.predict(texts)
print(categories['prediction'])
# ['TECH', 'TECH', 'OTHER']
```

#### Multiple Flags

```python
import re

# Multiline pattern with case insensitivity
log_clf = BinaryRegexClassifier(
    pattern=r'^ERROR',
    match_type='partial',
    flags=[re.IGNORECASE, re.MULTILINE],
    match_prediction='HIGH',
    no_match_prediction='LOW'
)

logs = pd.Series([
    'ERROR: System failure',
    'error: Connection lost',
    'Warning: Low memory'
])

severity = log_clf.predict(logs)
print(severity['prediction'])
# ['HIGH', 'HIGH', 'LOW']
```

#### With DataFrame

```python
# Specify column when using DataFrame
df = pd.DataFrame({
    'text': ['BUY NOW!!!', 'Meeting tomorrow', 'CLICK HERE'],
    'other': ['x', 'y', 'z']
})

spam_clf = BinaryRegexClassifier(
    pattern=r'\b(buy now|click here)\b',
    match_type='partial',
    ignore_case=True,
    column_name='text',
    match_prediction='SPAM',
    no_match_prediction='HAM'
)

result = spam_clf.predict(df)
print(result['prediction'])
# ['SPAM', 'HAM', 'SPAM']
```

### Match Types

#### Partial Match

Pattern found **anywhere** in the string:

```python
clf = BinaryRegexClassifier(pattern=r'\d{3}-\d{4}', match_type='partial')

data = pd.Series([
    'Call 123-4567',    # Match
    '123-4567',         # Match
    'No number'         # No match
])
```

#### Full Match

**Entire string** must match the pattern:

```python
clf = BinaryRegexClassifier(pattern=r'\d{3}-\d{4}', match_type='full')

data = pd.Series([
    'Call 123-4567',    # No match (extra text)
    '123-4567',         # Match
    'No number'         # No match
])
```

### Supported Flags

```python
import re

flags=[
    re.IGNORECASE,  # Case-insensitive
    re.MULTILINE,   # ^ and $ match line boundaries
    re.DOTALL,      # . matches newlines
    re.VERBOSE,     # Allow comments in pattern
    re.UNICODE      # Unicode matching
]
```

**Note:** Flags are automatically converted to inline Rust regex syntax for compatibility with the underlying regex engine.

### Common Patterns Reference

| Pattern | Regex | Description |
|---------|-------|-------------|
| Email | `r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}\b'` | Email addresses |
| URL | `r'https?://[^\s]+'` | HTTP/HTTPS URLs |
| Phone (US) | `r'\d{3}-\d{3}-\d{4}'` | Format: 123-456-7890 |
| Date | `r'\d{1,2}/\d{1,2}/\d{4}'` | Format: MM/DD/YYYY |
| IP Address | `r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'` | IPv4 addresses |
| Hashtag | `r'#\w+'` | Social media hashtags |
| Mention | `r'@\w+'` | Social media mentions |
| Credit Card | `r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'` | Card numbers (simple) |

### Use Cases

**Good for:**
- Quick baselines before ML
- Known patterns (emails, URLs, IDs)
- Data validation
- Log file filtering
- Content filtering with clear rules
- Pipeline integration (sklearn-compatible)

**Not ideal for:**
- Complex classification tasks
- Learning from data
- Nuanced pattern detection
- Ambiguous cases

### Integration with sklearn

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Use as feature generator
def add_regex_features(df):
    """Add regex-based features."""
    url_clf = BinaryRegexClassifier(pattern=r'https?://', match_type='partial')
    df['has_url'] = url_clf.predict(df['text'])['prediction']
    return df

# Or use in ensemble
# First try regex, fall back to ML if uncertain
```

### Performance

**Speed:** ⚡⚡⚡ Very fast (regex matching is highly optimized)  
**Memory:** Minimal (no training data stored)  
**Scalability:** Excellent (O(n) where n = number of samples)

### Best Practices

#### Do:
- Test patterns on sample data first
- Use raw strings (`r'pattern'`)
- Escape special characters properly
- Start simple, then refine
- Use online regex testers (regex101.com)

#### ❌ Don't:
- Make overly complex patterns
- Forget to handle edge cases
- Ignore performance on large datasets (though regex is generally fast)

### See Also

- [Tutorial: Regex Classifier](../../tutorials/regex_classifier.md) - In-depth guide
- [Getting Started](../../getting_started.md) - Quick start guide
- [BaseExplodingHamClassifier](../utils/base.md#baseexplodinghamclassifier) - Base classifier

---

**Note:** While this classifier doesn't "learn" from data, it's sklearn-compatible and useful for establishing baselines or implementing known rules in your ML pipeline.
