# Installation

## Requirements

- **Python**: 3.10 or higher
- **Dependencies**: 
  - `narwhals >= 2.14.0` (for DataFrame interoperability)
  - `setuptools >= 77.0.3` (for building)

## Install from PyPI

The easiest way to install ExplodingHam is using pip:

```bash
pip install explodingham
```

Note: The package name is `explodingham` (not `explodingham`).

## Install from Source

For development or to get the latest changes:

```bash
# Clone the repository
git clone https://github.com/pjachim/ExplodingHam.git
cd ExplodingHam

# Create a virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

## Verify Installation

Check that ExplodingHam is installed correctly:

```python
import explodingham
print(explodingham.__version__)  # Should print: 0.0.2

# Try importing a model
from explodingham.models.compression_learning.knn import CompressionKNN
print("✓ Installation successful!")
```

## Optional Dependencies
ExplodingHam uses narwhals for DataFrame operations, this makes it compatible with pandas or polars, but those are not automatically installed with ExplodingHam. As a result, you will want to install one of those two based on what worksfor your project.

### For Pandas Support
```bash
pip install pandas
```

### For Polars Support
```bash
pip install polars
```

### For Development
If you are developing new features for explodingham, you will also want to install pytest:
```bash
pip install pytest  # For running tests
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'explodingham'`:

1. Make sure you installed `explodingham`
2. Verify your virtual environment is activated
3. Try reinstalling: `pip uninstall explodingham && pip install explodingham`


## Next Steps

Now that ExplodingHam is installed, head to the [Getting Started Guide](getting_started.md) to build your first model!
