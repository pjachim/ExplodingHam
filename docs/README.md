# ExplodingHam Documentation

Complete documentation for the ExplodingHam library.

## Documentation Structure

📁 **docs/**
```
docs/
├── index.md                    # Documentation homepage
├── installation.md             # Installation guide
├── getting_started.md          # Quick start guide
├── architecture.md             # Design philosophy and structure
├── contributing.md             # Contributing guidelines
├── tutorials/                  # In-depth tutorials
│   ├── compression_knn.md      # Compression KNN guide
│   └── regex_classifier.md     # Regex classifier guide
├── examples/                   # Real-world examples
│   ├── text_classification.md  # Text classification examples
│   ├── dna_sequences.md        # DNA analysis examples
│   └── custom_compressors.md   # Custom compressor examples
└── api/                        # API reference
    ├── index.md                # API overview
    ├── models/
    │   ├── compression_learning.md
    │   └── baseline_models.md
    └── utils/
        ├── distance_metrics.md
        └── base.md
```

## Quick Navigation

### Getting Started
- [Installation](installation.md) - Install ExplodingHam
- [Getting Started](getting_started.md) - Your first model
- [Architecture](architecture.md) - How ExplodingHam works

### Learn
- [Compression KNN Tutorial](tutorials/compression_knn.md) - Deep dive into compression-based classification
- [Regex Classifier Tutorial](tutorials/regex_classifier.md) - Using regex for baselines
- [Text Classification Examples](examples/text_classification.md) - Practical applications
- [DNA Sequences Examples](examples/dna_sequences.md) - Bioinformatics use cases
- [Custom Compressors](examples/custom_compressors.md) - Build your own

### Reference
- [API Documentation](api/index.md) - Complete API reference
- [Contributing](contributing.md) - How to contribute

## Documentation Goals

This documentation aims to:

1. **Help you get started quickly** - Installation and first steps
2. **Teach core concepts** - Theory and best practices
3. **Provide real examples** - Copy-paste ready code
4. **Reference every feature** - Complete API documentation
5. **Enable contributions** - Clear development guidelines

## Documentation Style

- **Executable examples**: All code examples should run
- **Theory + Practice**: Explain the "why" and "how"
- **Multiple levels**: High-level guides + detailed API docs
- **Cross-references**: Links between related topics
- **Professional**: Ready for academic and industry use

## Building/Viewing Locally

Currently, documentation is in Markdown format. To view:

1. **VS Code**: Install "Markdown All in One" extension
2. **GitHub**: View online at the repository
3. **Static site** (future): Consider using MkDocs or Sphinx

## For Contributors

When adding features, update:

1. **Docstrings** in code (NumPy style)
2. **API reference** in `docs/api/`
3. **Tutorial** (for major features) in `docs/tutorials/`
4. **Examples** (show use cases) in `docs/examples/`
5. **Index pages** (add links to new content)

See [Contributing Guide](contributing.md) for details.

## Documentation TODOs

Future improvements:
- [ ] Set up MkDocs or Sphinx for professional site
- [ ] Add Jupyter notebook examples in docs
- [ ] Create video tutorials
- [ ] Add performance benchmarks
- [ ] Comparison tables with sklearn models
- [ ] FAQ section
- [ ] Troubleshooting guide

## Feedback

Found an issue with the documentation?
- [Open an issue](https://github.com/pjachim/ExplodingHam/issues)
- Suggest improvements
- Contribute fixes via PR

---

**Start here**: [Installation](installation.md) → [Getting Started](getting_started.md) → [Tutorials](tutorials/compression_knn.md)
