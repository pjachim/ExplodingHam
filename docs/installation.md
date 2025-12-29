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
print(explodingham.__version__)  # Should print: 0.0.1

# Try importing a model
from explodingham.models.compression_learning.knn import CompressionKNN
print("✓ Installation successful!")
```

## Optional Dependencies

Depending on your use case, you may want to install:

### For Pandas Support
```bash
pip install pandas
```

### For Polars Support
```bash
pip install polars
```

### For Development
```bash
pip install pytest  # For running tests
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'explodingham'`:

1. Make sure you installed `explodingham` (with an "i"), not `explodingham`
2. Verify your virtual environment is activated
3. Try reinstalling: `pip uninstall explodingham && pip install explodingham`

### Version Conflicts

If you encounter dependency conflicts:

```bash
# Create a fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
pip install explodingham
```

### Permission Errors on Windows

If you get permission errors during installation:

```bash
# Install for current user only
pip install --user explodingham
```

## Next Steps

Now that ExplodingHam is installed, head to the [Getting Started Guide](getting_started.md) to build your first model!
