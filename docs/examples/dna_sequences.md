# DNA Sequence Classification

Compression-based classification is particularly effective for biological sequence analysis. DNA sequences from similar organisms share structural patterns that compression algorithms can detect.

## Why Compression Works for DNA

DNA sequences have several properties that make them ideal for compression-based classification:

1. **Repetitive elements**: Tandem repeats, transposable elements
2. **Conserved regions**: Similar sequences indicate shared evolutionary history
3. **Structural patterns**: Coding regions, regulatory elements
4. **Limited alphabet**: Only 4 nucleotides (A, C, G, T)

When sequences share these patterns, they compress well together, giving low NCD (Normalized Compression Distance) values.

## Basic Organism Classification

### Bacteria vs Mammalian DNA

```python
from explodingham.models.compression_learning.knn import CompressionKNN
import pandas as pd

# Simulated DNA sequences
X_train = pd.DataFrame({
    'sequence': [
        # Bacterial sequences (simple, repetitive)
        'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG',
        'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT',
        'GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA',
        'GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC',
        
        # Mammalian sequences (more complex patterns)
        'ATGCATGCATGCGGGGCCCCATGCATGCATGCTTTTAAAATTTT',
        'TAGCTAGCTAGCAAAATTTTTAGCTAGCTAGCGGGGCCCCGGGG',
        'CGCGCGCGCGCGTACGTACGTACGTACGCGCGCGCGCGAAAATT',
        'ATATATATATGGGGCCCCTTTTAAAAATATATATGGGGCCCCTT',
    ]
})

y_train = pd.Series([
    'Bacteria', 'Bacteria', 'Bacteria', 'Bacteria',
    'Mammal', 'Mammal', 'Mammal', 'Mammal'
])

# Train classifier with LZMA (best for pattern detection)
dna_clf = CompressionKNN(
    k=3,
    data_column='sequence',
    compressor='lzma'  # Excellent for structured data
)

dna_clf.fit(X_train, y_train)

# Test sequences
X_test = pd.DataFrame({
    'sequence': [
        'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGGC',
        'ATGCATGCATGCGGGGCCCCTTTTAAAATAGCTAGCTAGCGGGG',
    ]
})

predictions = dna_clf.predict(X_test)

for seq, pred in zip(X_test['sequence'], predictions):
    print(f"{pred:8} | {seq[:40]}...")
```

**Output:**
```
Bacteria | ATCGATCGATCGATCGATCGATCGATCGATCGATCG...
Mammal   | ATGCATGCATGCGGGGCCCCTTTTAAAATAGCTAGC...
```

## Viral Classification

### Classifying Viral Families

```python
# Different viral DNA patterns
X_train = pd.DataFrame({
    'sequence': [
        # Herpesvirus (complex, large genome patterns)
        'ATGCATGCGGCCTATAGCTAGCATGCGGCCTATAGCTAGCATGC',
        'GGCCTATAGCTAGCATGCATGCGGCCTATAGCTAGCATGCATGC',
        
        # Parvovirus (smaller, simpler patterns)
        'ATATATATATATATATATAT',
        'GCGCGCGCGCGCGCGCGCGC',
        
        # Adenovirus (moderate complexity)
        'ATGCGCTAGGATCCATGCGCTAGGATCCATGCGCTAGGATCC',
        'GATCCATGCGCTAGGATCCATGCGCTAGGATCCATGCGCTAG',
    ]
})

y_train = pd.Series([
    'Herpesvirus', 'Herpesvirus',
    'Parvovirus', 'Parvovirus',
    'Adenovirus', 'Adenovirus'
])

viral_clf = CompressionKNN(k=2, data_column='sequence', compressor='lzma')
viral_clf.fit(X_train, y_train)

# Classify unknown viral sequences
X_test = pd.DataFrame({
    'sequence': [
        'ATATATATATATATAT',
        'ATGCATGCGGCCTATAGCTAGCATGC',
        'GATCCATGCGCTAGGATCCATGC'
    ]
})

viral_types = viral_clf.predict(X_test)
print("Viral classifications:", viral_types.to_list())
```

## Protein Coding Region Detection

### Identifying Coding vs Non-Coding Regions

```python
# Coding regions have specific patterns (codon structure)
X_train = pd.DataFrame({
    'sequence': [
        # Coding sequences (triplet structure, start/stop codons)
        'ATGTCAGGCCTAGCTTAAGGCATGCTA',
        'ATGGGCTTACCAGGATCCTAA',
        'ATGCCCGGGAAATTTCCCGGGTAA',
        
        # Non-coding sequences (more random, regulatory elements)
        'ACACACACACACACCCCGGGGTATATATA',
        'TTTTAAAATTTTAAAAGGGGCCCC',
        'CGCGCGCGTATATATACGCGCGCG',
    ]
})

y_train = pd.Series([
    'Coding', 'Coding', 'Coding',
    'Non-coding', 'Non-coding', 'Non-coding'
])

coding_clf = CompressionKNN(k=2, data_column='sequence', compressor='bz2')
coding_clf.fit(X_train, y_train)

# Test on new sequences
X_test = pd.DataFrame({
    'sequence': [
        'ATGGGCAAATTTCGGTAA',  # Has start codon ATG and stop codon TAA
        'TTTTTTTTAAAAAAAACCCCCCC'  # Random repeat
    ]
})

coding_status = coding_clf.predict(X_test)
print("Coding status:", coding_status.to_list())
# ['Coding', 'Non-coding']
```

## Species Identification

### Mitochondrial DNA Classification

```python
# Using mitochondrial DNA for species identification
X_train = pd.DataFrame({
    'mt_dna': [
        # Human mitochondrial patterns
        'GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCAT',
        'TTACTGCCAGCCACCATGAATATTGTACGGTACCATAAATACTTGACCAC',
        
        # Chimpanzee mitochondrial patterns  
        'GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTTCATGCAT',
        'TTACTGCCAGCCACCATGAATATTGTACGGTACCATAAACACTTGACCAC',
        
        # Gorilla mitochondrial patterns
        'GATCACAGGTCTATCACCCTATTAACCGCTCACGGGAGCTCTCCATGCAT',
        'TTACTGCCAGCCACCATGAATATTGTACGGTACCATAAATGCTTGACCAC',
    ]
})

y_train = pd.Series([
    'Human', 'Human',
    'Chimpanzee', 'Chimpanzee',
    'Gorilla', 'Gorilla'
])

species_clf = CompressionKNN(
    k=2,
    data_column='mt_dna',
    compressor='lzma'
)

species_clf.fit(X_train, y_train)

# Identify unknown sample
X_unknown = pd.DataFrame({
    'mt_dna': [
        'GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCAT'
    ]
})

species = species_clf.predict(X_unknown)
print(f"Species: {species[0]}")
```

## GC Content Classification

### High GC vs Low GC Organisms

```python
# Organisms have different GC content
X_train = pd.DataFrame({
    'sequence': [
        # High GC content (e.g., Streptomyces)
        'GCGCGCGCGCGCGGCCGGCCGGCCGCGCGCGC',
        'GGCCGGCCGGCCGCGCGCGCGCGGCCGGCCGC',
        'CGCGCGCGGCCGGCCGCGCGCGCGGCCGGCCG',
        
        # Low GC content (e.g., Plasmodium)
        'ATATATATATAT AAAAATTTTTAAAAATTTTT',
        'TTTTTAAAAATTTTTAAAAATATATATATATAT',
        'ATATATAAAAATTTTTATATATAAAAATTTTTA',
    ]
})

y_train = pd.Series([
    'High_GC', 'High_GC', 'High_GC',
    'Low_GC', 'Low_GC', 'Low_GC'
])

gc_clf = CompressionKNN(k=2, data_column='sequence', compressor='bz2')
gc_clf.fit(X_train, y_train)

# Classify based on GC content
X_test = pd.DataFrame({
    'sequence': [
        'GCGCGCGGCCGGCCGCGCGC',
        'ATATATAAAAATTTTTATATAT'
    ]
})

gc_classification = gc_clf.predict(X_test)
print("GC classifications:", gc_classification.to_list())
```

## Real-World Application: Metagenomic Binning

### Classifying Reads from Mixed Samples

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Simulated metagenomic reads (short sequences from environmental sample)
reads_df = pd.DataFrame({
    'read': [
        # E. coli reads
        'ATCGATCGATCG', 'GCTAGCTAGCTA', 'ATCGATCGATCGAT',
        # Bacillus reads
        'GGCCGGCCGGCC', 'CGCGCGCGCGCG', 'GGCCGGCCGGCCGG',
        # Pseudomonas reads
        'TACGTACGTACG', 'CGATCGATCGAT', 'TACGTACGTACGTA',
        # More samples...
    ] * 3,  # Repeat for more training data
    'organism': [
        'E.coli', 'E.coli', 'E.coli',
        'Bacillus', 'Bacillus', 'Bacillus',
        'Pseudomonas', 'Pseudomonas', 'Pseudomonas'
    ] * 3
})

X = reads_df[['read']]
y = reads_df['organism']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train metagenomic classifier
metagenomic_clf = CompressionKNN(
    k=3,
    data_column='read',
    compressor='lzma'
)

metagenomic_clf.fit(X_train, y_train)

# Classify reads
y_pred = metagenomic_clf.predict(X_test)

# Evaluate
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Performance Tips for DNA Data

### 1. Choose the Right Compressor

```python
# Test all compressors
for comp in ['gzip', 'bz2', 'lzma']:
    clf = CompressionKNN(k=3, data_column='sequence', compressor=comp)
    clf.fit(X_train, y_train)
    # Evaluate and compare
    print(f"Trained with {comp}")
```

**Recommendations:**
- **LZMA**: Best for complex patterns (eukaryotic DNA, large genomes)
- **BZ2**: Good balance for bacterial genomes
- **GZIP**: Fastest, good for simple repetitive sequences

### 2. Sequence Length Considerations

```python
# For very long sequences, consider windowing
def window_sequence(seq, window_size=50, step=25):
    """Break sequence into overlapping windows."""
    windows = []
    for i in range(0, len(seq) - window_size + 1, step):
        windows.append(seq[i:i + window_size])
    return windows

# Apply windowing
sequences = X_train['sequence'].apply(lambda x: window_sequence(x, 100, 50))
```

### 3. Handle Ambiguous Bases

```python
def clean_sequence(seq):
    """Remove or replace ambiguous bases (N, R, Y, etc.)."""
    # Simple approach: remove ambiguous bases
    clean = ''.join([base for base in seq if base in 'ACGT'])
    return clean

X_train['sequence'] = X_train['sequence'].apply(clean_sequence)
```

## Advanced: Reverse Complement Aware

```python
def reverse_complement(seq):
    """Get reverse complement of DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in reversed(seq)])

# Augment training data with reverse complements
X_augmented = pd.DataFrame({
    'sequence': list(X_train['sequence']) + 
                [reverse_complement(seq) for seq in X_train['sequence']]
})

y_augmented = pd.concat([y_train, y_train])

# Train on augmented data
clf_aug = CompressionKNN(k=3, data_column='sequence', compressor='lzma')
clf_aug.fit(X_augmented, y_augmented)
```

## Evaluation Metrics for DNA Classification

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Predict
y_pred = dna_clf.predict(X_test)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))
```

## Real Datasets

### Working with FASTA Files

```python
def read_fasta(filename):
    """Read sequences from FASTA file."""
    sequences = []
    labels = []
    current_seq = []
    current_label = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    labels.append(current_label)
                current_label = line[1:].split()[0]  # First word after >
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget last sequence
        if current_seq:
            sequences.append(''.join(current_seq))
            labels.append(current_label)
    
    return pd.DataFrame({'sequence': sequences}), pd.Series(labels)

# Load data
# X, y = read_fasta('sequences.fasta')

# Train
# clf = CompressionKNN(k=3, data_column='sequence', compressor='lzma')
# clf.fit(X, y)
```

## Limitations and Considerations

### When Compression Works Well:
- ✅ Clear structural differences between classes
- ✅ Sufficient sequence length (>20-30 bp)
- ✅ Moderate dataset sizes (< 10,000 sequences)
- ✅ Pattern-rich sequences

### When to Use Alternatives:
- ❌ Very short sequences (< 20 bp)
- ❌ Millions of sequences (too slow)
- ❌ When you need real-time classification
- ❌ Sequences are too similar (all from same species)

## Next Steps

- **[Text Classification Examples](text_classification.md)** - Apply similar concepts to text
- **[Custom Compressors](custom_compressors.md)** - Build specialized compressors
- **[API Reference](../api/models/compression_learning.md)** - Complete documentation

---

Compression-based classification provides a parameter-free, alignment-free approach to sequence analysis. It's particularly useful when traditional alignment-based methods are too slow or when dealing with novel sequences!
