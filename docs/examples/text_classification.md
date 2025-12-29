# Text Classification Examples

This guide demonstrates various text classification tasks using ExplodingHam's compression-based classifier.

## Language Detection

### Basic Language Classification

Classify text by language without any feature engineering:

```python
from explodingham.models.compression_learning.knn import CompressionKNN
import pandas as pd

# Training data with multiple languages
X_train = pd.DataFrame({
    'text': [
        'Hello world, how are you today?',
        'Good morning, nice to meet you',
        'Thank you very much for your help',
        'Hola mundo, cómo estás?',
        'Buenos días, mucho gusto',
        'Gracias por tu ayuda',
        'Bonjour le monde, comment allez-vous?',
        'Bonne journée, enchanté',
        'Merci beaucoup pour votre aide',
        'Ciao mondo, come stai?',
        'Buongiorno, piacere di conoscerla',
        'Grazie mille per il tuo aiuto',
    ]
})

y_train = pd.Series([
    'English', 'English', 'English',
    'Spanish', 'Spanish', 'Spanish',
    'French', 'French', 'French',
    'Italian', 'Italian', 'Italian'
])

# Create and train classifier
clf = CompressionKNN(
    k=3,  # Use 3 neighbors for robustness
    data_column='text',
    compressor='gzip'
)

clf.fit(X_train, y_train)

# Test on new data
X_test = pd.DataFrame({
    'text': [
        'Hi there, what is your name?',
        'Hola amigo, qué tal?',
        'Bonjour, comment ça va?',
        'Ciao, come va?'
    ]
})

predictions = clf.predict(X_test)

for text, pred in zip(X_test['text'], predictions):
    print(f"{pred:8} | {text}")
```

**Output:**
```
English  | Hi there, what is your name?
Spanish  | Hola amigo, qué tal?
French   | Bonjour, comment ça va?
Italian  | Ciao, come va?
```

### Multi-Language Dataset

Working with real multilingual data:

```python
# Load data (example structure)
df = pd.read_csv('multilingual_corpus.csv')
# Columns: 'text', 'language'

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[['text']], 
    df['language'],
    test_size=0.2,
    stratify=df['language'],
    random_state=42
)

# Train classifier
clf = CompressionKNN(k=5, data_column='text', compressor='gzip')
clf.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report, accuracy_score

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Sentiment Analysis

### Simple Sentiment Classification

```python
# Training data with positive/negative reviews
X_train = pd.DataFrame({
    'review': [
        'This product is absolutely amazing! I love it!',
        'Great quality, fast shipping, highly recommend',
        'Best purchase I ever made, 5 stars',
        'Terrible product, broke after one day',
        'Waste of money, very disappointed',
        'Poor quality, would not recommend to anyone',
        'Good product but a bit overpriced',
        'Works as expected, nothing special',
        'Excellent customer service and great product',
        'Horrible experience, never buying again'
    ]
})

y_train = pd.Series([
    'positive', 'positive', 'positive',
    'negative', 'negative', 'negative',
    'neutral', 'neutral',
    'positive', 'negative'
])

# Train sentiment classifier
sentiment_clf = CompressionKNN(
    k=3,
    data_column='review',
    compressor='bz2'  # BZ2 often works well for sentiment
)

sentiment_clf.fit(X_train, y_train)

# Classify new reviews
X_new = pd.DataFrame({
    'review': [
        'Amazing quality, will buy again!',
        'Not worth the price, disappointed',
        'It works fine, nothing remarkable'
    ]
})

sentiments = sentiment_clf.predict(X_new)

for review, sentiment in zip(X_new['review'], sentiments):
    print(f"[{sentiment.upper():8}] {review}")
```

## Spam Detection

### Email Spam Classification

```python
X_train = pd.DataFrame({
    'email': [
        'Meeting scheduled for tomorrow at 3pm',
        'Project update: milestone completed',
        'Lunch on Friday?',
        'Can you review this document?',
        'CONGRATULATIONS!!! You WON $1,000,000!!!',
        'BUY NOW!!! LIMITED TIME OFFER!!!',
        'Click here for FREE MONEY!!!',
        'You are the LUCKY WINNER!!!',
        'Reminder: team standup at 10am',
        'URGENT: Claim your prize NOW!!!',
        'Weekly report attached',
        'FREE VIAGRA!!! CLICK HERE!!!'
    ]
})

y_train = pd.Series([
    'ham', 'ham', 'ham', 'ham',
    'spam', 'spam', 'spam', 'spam',
    'ham', 'spam', 'ham', 'spam'
])

# Train spam detector
spam_clf = CompressionKNN(
    k=3,
    data_column='email',
    compressor='gzip'
)

spam_clf.fit(X_train, y_train)

# Test on new emails
X_test = pd.DataFrame({
    'email': [
        'Conference call details for next week',
        'WIN A FREE IPHONE NOW!!!',
        'Please find the quarterly report attached',
        'CLICK HERE FOR AMAZING DEALS!!!'
    ]
})

classifications = spam_clf.predict(X_test)

for email, classification in zip(X_test['email'], classifications):
    emoji = '📧' if classification == 'ham' else '🚫'
    print(f"{emoji} [{classification.upper():4}] {email[:50]}...")
```

## Topic Classification

### News Article Classification

```python
# News articles by topic
X_train = pd.DataFrame({
    'article': [
        'The stock market reached new highs today as investors...',
        'Central bank announces interest rate decision...',
        'Tech company reports quarterly earnings...',
        'Scientists discover new exoplanet in distant galaxy...',
        'Research team develops breakthrough cancer treatment...',
        'Climate study reveals concerning trends...',
        'Championship game ends in dramatic finish...',
        'Athlete breaks world record in Olympic event...',
        'Local team wins tournament after comeback...',
    ]
})

y_train = pd.Series([
    'business', 'business', 'business',
    'science', 'science', 'science',
    'sports', 'sports', 'sports'
])

topic_clf = CompressionKNN(
    k=2,
    data_column='article',
    compressor='lzma'  # LZMA for more pattern detection
)

topic_clf.fit(X_train, y_train)

# Classify new articles
X_test = pd.DataFrame({
    'article': [
        'Startup raises $50M in funding round...',
        'New telescope captures stunning images of nebula...',
        'Team advances to playoffs after victory...'
    ]
})

topics = topic_clf.predict(X_test)
print("Article Topics:", topics.to_list())
# ['business', 'science', 'sports']
```

## Author Attribution

### Identifying Writing Style

```python
# Classic literature samples
X_train = pd.DataFrame({
    'text': [
        "It was the best of times, it was the worst of times...",
        "It is a far, far better thing that I do...",
        "Call me Ishmael. Some years ago...",
        "Moby Dick seeks thee not. It is thou...",
        "It is a truth universally acknowledged that a single man...",
        "There is a stubbornness about me that never can bear...",
        "In a hole in the ground there lived a hobbit...",
        "All we have to decide is what to do with the time...",
    ]
})

y_train = pd.Series([
    'Dickens', 'Dickens',
    'Melville', 'Melville',
    'Austen', 'Austen',
    'Tolkien', 'Tolkien'
])

author_clf = CompressionKNN(
    k=1,  # k=1 for distinctive styles
    data_column='text',
    compressor='lzma'  # Best for style detection
)

author_clf.fit(X_train, y_train)

# Test on new passages
X_test = pd.DataFrame({
    'text': [
        "It was a bright cold day in April...",  # Orwell (not in training)
        "Not all those who wander are lost...",  # Tolkien
    ]
})

authors = author_clf.predict(X_test)
print("Predicted authors:", authors.to_list())
```

## Programming Language Detection

### Code Snippet Classification

```python
X_train = pd.DataFrame({
    'code': [
        'def hello():\n    print("Hello")',
        'for i in range(10):\n    print(i)',
        'import pandas as pd\ndf = pd.DataFrame()',
        'function hello() {\n    console.log("Hello");\n}',
        'const arr = [1, 2, 3];\narr.forEach(x => console.log(x));',
        'let x = {name: "John", age: 30};',
        '#include <stdio.h>\nint main() {\n    printf("Hello");\n}',
        'void swap(int *a, int *b) {\n    int temp = *a;\n}',
        'int arr[] = {1, 2, 3, 4, 5};',
    ]
})

y_train = pd.Series([
    'Python', 'Python', 'Python',
    'JavaScript', 'JavaScript', 'JavaScript',
    'C', 'C', 'C'
])

code_clf = CompressionKNN(
    k=2,
    data_column='code',
    compressor='bz2'
)

code_clf.fit(X_train, y_train)

# Test on new code
X_test = pd.DataFrame({
    'code': [
        'print("test")\nfor x in list:\n    print(x)',
        'console.log("test");\nconst y = 10;',
        'printf("test");\nint x = 10;'
    ]
})

languages = code_clf.predict(X_test)
print("Detected languages:", languages.to_list())
# ['Python', 'JavaScript', 'C']
```

## Intent Classification (Chatbot)

### User Intent Detection

```python
# Training data with user intents
X_train = pd.DataFrame({
    'message': [
        'What is the weather today?',
        'Will it rain tomorrow?',
        'How cold is it outside?',
        'Book a flight to New York',
        'Reserve a table for two',
        'Schedule a meeting for 3pm',
        'What is 15 plus 27?',
        'Calculate the square root of 144',
        'How many days until Christmas?',
        'Tell me a joke',
        'What is the meaning of life?',
        'Sing a song for me',
    ]
})

y_train = pd.Series([
    'weather', 'weather', 'weather',
    'booking', 'booking', 'booking',
    'calculation', 'calculation', 'calculation',
    'entertainment', 'entertainment', 'entertainment'
])

intent_clf = CompressionKNN(
    k=3,
    data_column='message',
    compressor='gzip'
)

intent_clf.fit(X_train, y_train)

# Classify user messages
X_test = pd.DataFrame({
    'message': [
        'Is it going to snow?',
        'Book a hotel room',
        'What is 100 divided by 5?',
        'Make me laugh'
    ]
})

intents = intent_clf.predict(X_test)

for msg, intent in zip(X_test['message'], intents):
    print(f"[{intent:12}] {msg}")
```

## Tips for Better Text Classification

### 1. Data Preprocessing

```python
import pandas as pd

def preprocess_text(text):
    """Clean text before classification."""
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove URLs (optional)
    import re
    text = re.sub(r'http\S+', '', text)
    
    return text

# Apply preprocessing
X_train['text'] = X_train['text'].apply(preprocess_text)
X_test['text'] = X_test['text'].apply(preprocess_text)
```

### 2. Choosing k

```python
from sklearn.model_selection import cross_val_score

# Test different k values
for k in [1, 3, 5, 7]:
    clf = CompressionKNN(k=k, data_column='text')
    # scores = cross_val_score(clf, X, y, cv=5)
    # print(f"k={k}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 3. Compressor Selection

```python
# Test different compressors
for comp in ['gzip', 'bz2', 'lzma']:
    clf = CompressionKNN(k=3, data_column='text', compressor=comp)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test) if hasattr(clf, 'score') else None
    print(f"{comp}: ready")
```

### 4. Handling Imbalanced Data

```python
from sklearn.utils import resample

# Upsample minority class
df_majority = df[df['label'] == 'majority_class']
df_minority = df[df['label'] == 'minority_class']

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
```

## Evaluation Metrics

### Comprehensive Evaluation

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Next Steps

- **[DNA Sequence Classification](dna_sequences.md)** - Bioinformatics applications
- **[Custom Compressors](custom_compressors.md)** - Build your own compression functions
- **[API Reference](../api/models/compression_learning.md)** - Complete documentation

---

Compression-based classification works remarkably well for text without any feature engineering. Experiment with different compressors and k values to optimize for your specific task!
