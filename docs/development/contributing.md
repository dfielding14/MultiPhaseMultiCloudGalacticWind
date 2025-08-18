# Contributing Guide

## Overview

We welcome contributions to the Multiphase Galactic Wind model! This guide explains how to contribute effectively.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/GalacticWindsMultiphaseAnalytic.git
cd GalacticWindsMultiphaseAnalytic
pip install -e .
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Development Setup

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Check code style
flake8 multiphasegalacticwind/
```

## Code Standards

### Style Guide

We follow PEP 8 with these additions:
- Line length: 100 characters
- Docstrings: NumPy style
- Type hints: Encouraged

### Documentation

All functions need docstrings:

```python
def calculate_cooling_time(T: float, n: float, Z: float = 1.0) -> float:
    """
    Calculate radiative cooling time.
    
    Parameters
    ----------
    T : float
        Temperature [K]
    n : float
        Number density [cm^-3]
    Z : float, optional
        Metallicity [solar units]
        
    Returns
    -------
    float
        Cooling time [s]
    """
```

### Testing

Write tests for new features:

```python
def test_new_feature():
    """Test the new feature works correctly."""
    result = new_feature(input_data)
    assert result == expected_value
```

## Making Changes

### 1. Physics Changes

When modifying physics:
- Document equations in LaTeX
- Add references to papers
- Include validation tests
- Update relevant examples

### 2. API Changes

For API modifications:
- Maintain backward compatibility
- Update documentation
- Add deprecation warnings
- Update examples

### 3. Performance Changes

For optimization:
- Benchmark before/after
- Document improvements
- Ensure accuracy maintained
- Add performance tests

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] Commit messages clear
- [ ] Branch up to date

### 2. PR Description

Include:
- What changes were made
- Why they were needed
- How they were tested
- Any breaking changes

### 3. Review Process

- Automated tests run
- Code review by maintainer
- Address feedback
- Merge when approved

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Tests

```bash
pytest tests/test_wind_model.py::test_specific
```

### Coverage Report

```bash
pytest --cov=multiphasegalacticwind tests/
```

## Documentation

### Build Docs Locally

```bash
mkdocs serve
# Visit http://127.0.0.1:8000
```

### Add Documentation

1. Create markdown files in `docs/`
2. Update `mkdocs.yml` navigation
3. Use proper cross-references
4. Include code examples

## Common Tasks

### Adding a Parameter

1. Add to `WindConfig.__init__()`
2. Add validation in `validate()`
3. Add default in `_custom_defaults`
4. Update documentation
5. Add tests

### Adding an Observable

1. Implement in `observables.py`
2. Add to `WindSolution` class
3. Create plotting function
4. Document usage
5. Add example

### Fixing a Bug

1. Create failing test
2. Fix the bug
3. Verify test passes
4. Update changelog
5. Submit PR

## Development Guidelines

From `DEVELOPMENT_GUIDELINES.md`:
- Think critically, push back on poor ideas
- Consider multiple approaches
- Maximum 3 attempts per issue
- Small incremental changes
- Simplicity over cleverness

## Release Process

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create git tag
4. Push to GitHub
5. Create release

## Getting Help

- Open an issue for bugs
- Start a discussion for features
- Check existing issues first
- Provide minimal examples

## Code of Conduct

- Be respectful
- Welcome newcomers
- Focus on what's best for the project
- Accept constructive criticism
- Show empathy

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Acknowledgments

Contributors will be acknowledged in:
- Git history
- CONTRIBUTORS.md
- Release notes
- Papers (for significant contributions)