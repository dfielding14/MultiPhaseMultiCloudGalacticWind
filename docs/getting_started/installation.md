# Installation

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## Install from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/GalacticWindsMultiphaseAnalytic.git
cd GalacticWindsMultiphaseAnalytic
pip install -e .
```

## Dependencies

### Required
```bash
pip install numpy scipy matplotlib h5py
```

### Optional (for full features)
```bash
pip install cmasher  # Better colormaps
pip install jupyter  # For notebooks
pip install pytest   # For testing
```

## Verify Installation

Test that the installation worked:

```python
from multiphasegalacticwind import WindModel

# Create a simple model
model = WindModel(SFR=1.0)
print("Installation successful!")
```

## Troubleshooting

### ImportError
If you get import errors, ensure the package is in your Python path:
```python
import sys
sys.path.append('/path/to/GalacticWindsMultiphaseAnalytic')
```

### Cooling Tables
The package includes pre-computed cooling tables from Wiersma+09. These are automatically loaded from the `data/` directory.

### LaTeX Rendering
For publication-quality plots with LaTeX labels, install:
```bash
# Ubuntu/Debian
sudo apt-get install texlive texlive-latex-extra dvipng

# macOS
brew install --cask mactex
```