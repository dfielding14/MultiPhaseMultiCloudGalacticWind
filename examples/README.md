# Examples

This directory contains examples demonstrating various features of the `multiphasegalacticwind` package. Each example serves a specific purpose:

## Quick Start
- **`simple_example.py`** - Minimal working example (45 lines)
  - Quick test to verify installation
  - Basic model creation and execution
  - Simple plots saved as PDFs

## Research Applications
- **`comprehensive_example.py`** - Production-quality research example (228 lines)
  - Specific parameters: SFR=20, η_M=0.1, η_M_cold=0.1, 5 cloud masses
  - Full analysis workflow with multiple plots
  - Parameter study varying cold mass loading
  - Saves all plots to `plots/` directory

## Observational Analysis
- **`column_density_example.py`** - Column density calculations (155 lines)
  - Specialized for absorption line observations
  - Shows dN/dv in cm^-2 / (km/s) units
  - Individual cloud species contributions
  - Parameter studies for observational predictions

## Interactive Tutorials
- **`tutorial_comprehensive.ipynb`** - Complete interactive tutorial
  - Step-by-step introduction to the package
  - Detailed explanations of physics
  - Interactive parameter exploration
  - Best starting point for new users

- **`observational_comparison_notebook.ipynb`** - Observatory-focused analysis
  - Mock absorption line profiles
  - M82-like starburst galaxy example
  - Velocity distribution analysis
  - For observers planning observations

## Running the Examples

```bash
# Python scripts
python simple_example.py
python comprehensive_example.py
python column_density_example.py

# Jupyter notebooks
jupyter notebook tutorial_comprehensive.ipynb
```

All examples save plots as PDF files rather than displaying them interactively to avoid hanging.