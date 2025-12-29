# Regex Classifier Tutorial

## Introduction

The `BinaryRegexClassifier` is a non-ML baseline classifier that uses regular expressions to classify data. While it doesn't "learn" from data, it's incredibly useful for:

- **Quick baselines**: Test your ML model against simple rules
- **Known patterns**: When you have clear rules (emails, URLs, etc.)
- **Pipeline integration**: Drop-in sklearn-compatible classifier
- **Experimentation**: Rapidly test different regex patterns

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Match Types: Partial vs Full](#match-types)
3. [Regex Flags and Case Sensitivity](#flags)
4. [Custom Predictions](#custom-predictions)
5. [Real-World Examples](#examples)
6. [Advanced Patterns](#advanced)
7. [Integration with sklearn](#integration)

## <a name="basic-usage"></a>Basic Usage

### Simple URL Detection

```python
from explodingham.models.baseline_models.regex import BinaryRegexClassifier
import pandas as pd

# Create classifier to detect URLs
url_clf = BinaryRegexClassifier(
    pattern=r'https?://',
    match_type='partial'
)

# Test data
urls = pd.Series([
    'https://example.com',
    'http://site.org',
    'ftp://files.com',
    'example.com',
    'https://secure.bank.com/login'
])

# Predict
predictions = url_clf.predict(urls)
print(predictions['prediction'])
# [1, 1, 0, 0, 1]
```

### Email Validation

```python
# Detect email addresses
email_clf = BinaryRegexClassifier(
    pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    match_type='partial'
)

emails = pd.Series([
    'support@example.com',
    'not an email',
    'user@domain.co.uk'
])

results = email_clf.predict(emails)['prediction']
# [1, 0, 1]
```

## <a name="match-types"></a>Match Types: Partial vs Full

### Partial Match

**Pattern found anywhere in the string:**

```python
partial_clf = BinaryRegexClassifier(
    pattern=r'\d{3}-\d{4}',  # Phone number pattern
    match_type='partial'
)

data = pd.Series([
    'Call me at 123-4567',     # Match (contains pattern)
    '123-4567',                # Match (is pattern)
    'No number here',          # No match
    'Dial 999-0000 now'        # Match (contains pattern)
])

results = partial_clf.predict(data)['prediction']
# [1, 1, 0, 1]
```

### Full Match

**Entire string must match the pattern:**

```python
full_clf = BinaryRegexClassifier(
    pattern=r'\d{3}-\d{4}',
    match_type='full'
)

data = pd.Series([
    'Call me at 123-4567',     # No match (extra text)
    '123-4567',                # Match (exact)
    'No number here',          # No match
    '999-0000'                 # Match (exact)
])

results = full_clf.predict(data)['prediction']
# [0, 1, 0, 1]
```

### When to Use Each

**Use `match_type='partial'`:**
- Detecting presence of patterns (URLs in text, email mentions)
- Filtering log files
- Finding keywords or entities

**Use `match_type='full'`:**
- Validating formats (phone numbers, ZIP codes)
- Classifying based on structure
- Ensuring strict compliance

## <a name="flags"></a>Regex Flags and Case Sensitivity

### Ignore Case

```python
# Case-insensitive matching
case_clf = BinaryRegexClassifier(
    pattern=r'python',
    match_type='partial',
    ignore_case=True  # Convenience parameter
)

texts = pd.Series([
    'I love Python',
    'PYTHON rocks',
    'python is great',
    'Java is nice'
])

results = case_clf.predict(texts)['prediction']
# [1, 1, 1, 0]
```

### Using Multiple Flags

```python
import re

# Multiline pattern with case insensitivity
multi_clf = BinaryRegexClassifier(
    pattern=r'^ERROR',  # Match ERROR at line start
    match_type='partial',
    flags=[re.IGNORECASE, re.MULTILINE]
)

logs = pd.Series([
    'ERROR: System failure',
    'error: Connection lost',
    'Warning: Low memory',
    'Line 1\nerror: Failed'  # Multiline string
])

results = multi_clf.predict(logs)['prediction']
# [1, 1, 0, 1]
```

### Supported Flags

```python
import re

clf = BinaryRegexClassifier(
    pattern=r'...',
    flags=[
        re.IGNORECASE,  # Case-insensitive
        re.MULTILINE,   # ^ and $ match line boundaries
        re.DOTALL,      # . matches newlines
        re.VERBOSE,     # Allow comments in pattern
        re.UNICODE      # Unicode matching
    ]
)
```

## <a name="custom-predictions"></a>Custom Predictions

### Custom Labels

```python
# Return descriptive labels instead of 0/1
sentiment_clf = BinaryRegexClassifier(
    pattern=r'\b(great|awesome|excellent|love)\b',
    match_type='partial',
    ignore_case=True,
    match_prediction='POSITIVE',
    no_match_prediction='NEUTRAL'
)

reviews = pd.Series([
    'This product is great!',
    'Average quality',
    'I love it!',
    'It works'
])

results = sentiment_clf.predict(reviews)['prediction']
# ['POSITIVE', 'NEUTRAL', 'POSITIVE', 'NEUTRAL']
```

### Custom Column Name

```python
# Custom prediction column name
clf = BinaryRegexClassifier(
    pattern=r'error',
    match_type='partial',
    prediction_name='status',  # Default is 'prediction'
    match_prediction='ERROR',
    no_match_prediction='OK'
)

logs = pd.Series(['error in line 10', 'success'])
results = clf.predict(logs)

print(results['status'])  # Access via custom name
# ['ERROR', 'OK']
```

## <a name="examples"></a>Real-World Examples

### Log File Classification

```python
# Classify log severity
log_clf = BinaryRegexClassifier(
    pattern=r'\b(ERROR|CRITICAL|FATAL)\b',
    match_type='partial',
    ignore_case=True,
    match_prediction='HIGH_SEVERITY',
    no_match_prediction='LOW_SEVERITY'
)

logs = pd.DataFrame({
    'message': [
        'INFO: Application started',
        'ERROR: Database connection failed',
        'DEBUG: Processing request',
        'CRITICAL: Out of memory'
    ]
})

logs['severity'] = log_clf.predict(logs['message'])['prediction']
print(logs)
```

### Data Validation

```python
# Validate phone numbers
phone_clf = BinaryRegexClassifier(
    pattern=r'^\d{3}-\d{3}-\d{4}$',
    match_type='full',
    match_prediction='VALID',
    no_match_prediction='INVALID'
)

phone_numbers = pd.Series([
    '123-456-7890',
    '999-888-7777',
    '123-45-6789',  # Wrong format
    'not a phone',
    '111-222-3333'
])

validation = phone_clf.predict(phone_numbers)['prediction']
# ['VALID', 'VALID', 'INVALID', 'INVALID', 'VALID']
```

### Content Filtering

```python
# Filter spam keywords
spam_clf = BinaryRegexClassifier(
    pattern=r'\b(buy now|click here|free money|congratulations|winner)\b',
    match_type='partial',
    ignore_case=True,
    match_prediction='SPAM',
    no_match_prediction='HAM'
)

emails = pd.DataFrame({
    'subject': [
        'Meeting tomorrow at 3pm',
        'BUY NOW! Limited offer!!!',
        'Project update',
        'Congratulations! You are a winner!'
    ]
})

emails['classification'] = spam_clf.predict(emails['subject'])['prediction']
```

### Language Detection (Simple)

```python
# Detect Spanish (very simple heuristic)
spanish_clf = BinaryRegexClassifier(
    pattern=r'\b(el|la|los|las|de|en|un|una|es|está|por|para)\b',
    match_type='partial',
    match_prediction='Spanish',
    no_match_prediction='Other'
)

texts = pd.Series([
    'El gato es negro',
    'The cat is black',
    'La casa está en la ciudad',
    'Hello world'
])

language = spanish_clf.predict(texts)['prediction']
# ['Spanish', 'Other', 'Spanish', 'Other']
```

## <a name="advanced"></a>Advanced Patterns

### Capture Groups (Indirect Usage)

While the classifier doesn't expose capture groups directly, you can design patterns strategically:

```python
# Match dates in various formats
date_clf = BinaryRegexClassifier(
    pattern=r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
    match_type='partial'
)

texts = pd.Series([
    'Meeting on 12/25/2023',
    'Due date: 2023-12-25',
    'Tomorrow',
    'Event on 1-15-24'
])

has_date = date_clf.predict(texts)['prediction']
# [1, 0, 0, 1]  # Second one doesn't match our pattern
```

### Complex Patterns

```python
# Detect credit card numbers (simple check)
cc_clf = BinaryRegexClassifier(
    pattern=r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
    match_type='partial',
    match_prediction='REDACT',
    no_match_prediction='OK'
)

texts = pd.Series([
    'Card: 1234-5678-9012-3456',
    'Card: 1234 5678 9012 3456',
    'No card here',
    'Card: 1234567890123456'
])

security = cc_clf.predict(texts)['prediction']
# ['REDACT', 'REDACT', 'OK', 'REDACT']
```

### Alternation

```python
# Match multiple programming languages
prog_lang_clf = BinaryRegexClassifier(
    pattern=r'\b(python|javascript|java|ruby|go|rust)\b',
    match_type='partial',
    ignore_case=True,
    match_prediction='TECH',
    no_match_prediction='NON_TECH'
)

texts = pd.Series([
    'I love Python programming',
    'JavaScript is versatile',
    'Cooking is fun',
    'Learning Rust'
])

category = prog_lang_clf.predict(texts)['prediction']
# ['TECH', 'TECH', 'NON_TECH', 'TECH']
```

## <a name="integration"></a>Integration with sklearn

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Use regex as first filter, ML as fallback
class RegexOrML:
    def __init__(self, regex_clf, ml_clf):
        self.regex_clf = regex_clf
        self.ml_clf = ml_clf
    
    def fit(self, X, y):
        self.ml_clf.fit(X, y)
        return self
    
    def predict(self, X):
        # Try regex first
        regex_results = self.regex_clf.predict(X)['prediction']
        
        # For uncertain cases, use ML
        # (This is a simple example - refine for production)
        return regex_results
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin

# Wrap for sklearn compatibility
class SklearnRegexClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, pattern, match_type='partial', ignore_case=False):
        self.pattern = pattern
        self.match_type = match_type
        self.ignore_case = ignore_case
        self.clf = BinaryRegexClassifier(
            pattern=pattern,
            match_type=match_type,
            ignore_case=ignore_case
        )
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return self.clf.predict(X)['prediction'].to_numpy()

# Now you can use it with sklearn tools
clf = SklearnRegexClassifier(pattern=r'spam', ignore_case=True)
# scores = cross_val_score(clf, X, y, cv=5)
```

### As a Feature Generator

```python
import pandas as pd

# Use regex matches as features
def add_regex_features(df, text_column):
    """Add regex-based features to dataframe."""
    
    # Has URL
    url_clf = BinaryRegexClassifier(pattern=r'https?://', match_type='partial')
    df['has_url'] = url_clf.predict(df[text_column])['prediction']
    
    # Has email
    email_clf = BinaryRegexClassifier(
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        match_type='partial'
    )
    df['has_email'] = email_clf.predict(df[text_column])['prediction']
    
    # Has number
    number_clf = BinaryRegexClassifier(pattern=r'\d+', match_type='partial')
    df['has_number'] = number_clf.predict(df[text_column])['prediction']
    
    return df

# Use features for ML
texts = pd.DataFrame({
    'text': [
        'Visit https://example.com',
        'Email me at user@example.com',
        'Just regular text',
        'Call 123-4567'
    ]
})

texts_with_features = add_regex_features(texts, 'text')
print(texts_with_features)
```

## Best Practices

### ✅ Do:
- Test patterns on sample data first
- Use raw strings (`r'pattern'`) for regex
- Escape special characters (`\.`, `\?`, etc.)
- Start with simple patterns, then refine
- Use online regex testers (regex101.com)

### ❌ Don't:
- Expect regex to replace ML for complex tasks
- Use overly complex patterns (hard to maintain)
- Forget to handle edge cases
- Ignore performance on large datasets

## Debugging Patterns

### Test in Python REPL

```python
import re

pattern = r'\d{3}-\d{4}'
test_strings = ['123-4567', 'abc-defg', '99-00']

for s in test_strings:
    if re.search(pattern, s):
        print(f"✓ '{s}' matches")
    else:
        print(f"✗ '{s}' doesn't match")
```

### Use Verbose Mode

```python
import re

# Complex pattern with comments
pattern = r'''
    ^                 # Start of string
    \d{3}             # Three digits
    [-]               # Literal hyphen
    \d{4}             # Four digits
    $                 # End of string
'''

clf = BinaryRegexClassifier(
    pattern=pattern,
    match_type='full',
    flags=[re.VERBOSE]
)
```

## Common Patterns Reference

### Email
```python
pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
```

### URL
```python
pattern=r'https?://[^\s]+'
```

### Phone (US)
```python
pattern=r'\d{3}-\d{3}-\d{4}'
```

### Date (MM/DD/YYYY)
```python
pattern=r'\d{1,2}/\d{1,2}/\d{4}'
```

### IP Address
```python
pattern=r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
```

### Hashtag
```python
pattern=r'#\w+'
```

### Mention
```python
pattern=r'@\w+'
```

## Next Steps

- **[API Reference](../api/models/baseline_models.md)** - Complete parameter docs
- **[Getting Started Guide](../getting_started.md)** - More examples
- **[Compression KNN Tutorial](compression_knn.md)** - Learn about actual ML in ExplodingHam

---

The `BinaryRegexClassifier` is perfect for quick baselines and known patterns. Use it to establish performance benchmarks before investing in more complex ML solutions!
