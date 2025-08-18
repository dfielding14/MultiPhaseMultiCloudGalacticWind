# Testing Guide

## Overview

Comprehensive testing ensures the physics implementation is correct and numerical methods are stable.

## Test Structure

```
tests/
├── test_wind_model.py       # High-level model tests
├── test_core_physics.py     # ODE solver tests
├── test_config.py           # Configuration validation
├── test_observables.py      # Observable calculations
├── test_cooling.py          # Cooling function tests
├── test_fixes.py            # Bug fix validation
└── test_performance.py      # Performance benchmarks
```

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### Specific Module

```bash
pytest tests/test_wind_model.py -v
```

### With Coverage

```bash
pytest --cov=multiphasegalacticwind tests/
```

### Parallel Execution

```bash
pytest -n auto tests/
```

## Test Categories

### 1. Unit Tests

Test individual functions:

```python
def test_cooling_time():
    """Test cooling time calculation."""
    from multiphasegalacticwind.cooling import tcool_P
    
    T = 1e6  # K
    P_over_kB = 1000  # K/cm^3
    Z = 1.0
    
    t_cool = tcool_P(T, P_over_kB, Z, 0.0, 0.62)
    
    # Check reasonable value (~ Myr)
    assert 1e13 < t_cool < 1e15
```

### 2. Integration Tests

Test component interactions:

```python
def test_wind_model_integration():
    """Test full model integration."""
    from multiphasegalacticwind import WindModel
    
    model = WindModel(SFR=10.0, v_circ=200.0)
    solution = model.run()
    
    # Check solution properties
    assert solution.v_terminal > 0
    assert solution.r[-1] > 10  # Reaches > 10 kpc
    assert np.all(solution.v > 0)  # Positive velocities
```

### 3. Physics Tests

Validate physics implementation:

```python
def test_energy_conservation():
    """Test energy conservation in hot-only wind."""
    model = WindModel(SFR=10.0)
    solution = model.run_hot_only()
    
    # Calculate Bernoulli parameter
    B = 0.5 * solution.v**2 + 2.5 * solution.P / solution.rho
    
    # Should be approximately constant
    assert np.std(B) / np.mean(B) < 0.01
```

### 4. Regression Tests

Ensure results don't change unexpectedly:

```python
def test_regression_terminal_velocity():
    """Test terminal velocity hasn't changed."""
    model = WindModel(
        SFR=10.0,
        v_circ=200.0,
        eta_M=0.1,
        eta_E=1.0
    )
    solution = model.run()
    
    # Known value from previous version
    expected = 485.3  # km/s
    assert abs(solution.v_terminal - expected) < 1.0
```

### 5. Numerical Tests

Test numerical stability:

```python
def test_sonic_point_regularization():
    """Test regularization near Mach = 1."""
    from multiphasegalacticwind.core_physics import Wind_Evo
    
    # Create state near sonic point
    v_wind = 300e5  # cm/s
    cs = 300e5      # Same as v -> Mach = 1
    rho = 1e-26
    P = rho * cs**2 / 1.67
    
    state = [v_wind, rho, P, rho, 1e4, v_wind, 1.0]
    r = 1e20  # 1 kpc in cm
    
    # Should not crash or return NaN
    derivs = Wind_Evo(r, state, params)
    assert np.all(np.isfinite(derivs))
```

## Performance Testing

### Benchmarking

```python
import time

def test_model_performance():
    """Benchmark model execution time."""
    model = WindModel(SFR=10.0, rtol=1e-6)
    
    start = time.time()
    solution = model.run()
    elapsed = time.time() - start
    
    print(f"Model run time: {elapsed:.2f} s")
    assert elapsed < 10.0  # Should complete in < 10s
```

### Profiling

```python
import cProfile

def profile_wind_evolution():
    """Profile wind evolution performance."""
    model = WindModel(SFR=10.0)
    
    profiler = cProfile.Profile()
    profiler.enable()
    solution = model.run()
    profiler.disable()
    
    profiler.print_stats(sort='cumulative', lines=20)
```

## Test Fixtures

### Common Setup

```python
import pytest

@pytest.fixture
def standard_model():
    """Create standard model for testing."""
    from multiphasegalacticwind import WindModel
    return WindModel(SFR=10.0, v_circ=200.0)

@pytest.fixture
def galaxy_data():
    """Load test galaxy data."""
    return {
        'name': 'TestGalaxy',
        'sfr': 10.0,
        'v_circ': 200.0,
        'ions': {...}
    }
```

### Using Fixtures

```python
def test_with_fixture(standard_model):
    """Test using fixture."""
    solution = standard_model.run()
    assert solution.v_terminal > 0
```

## Continuous Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ -v --cov=multiphasegalacticwind
```

## Test Data

### Generate Test Data

```python
def generate_test_solution():
    """Generate reference solution for tests."""
    model = WindModel(SFR=10.0)
    solution = model.run()
    
    # Save for regression tests
    np.savez('tests/data/reference_solution.npz',
             r=solution.r,
             v=solution.v,
             rho=solution.rho,
             P=solution.P)
```

### Load Test Data

```python
def test_against_reference():
    """Test against reference solution."""
    # Load reference
    ref = np.load('tests/data/reference_solution.npz')
    
    # Run model
    model = WindModel(SFR=10.0)
    solution = model.run()
    
    # Compare
    np.testing.assert_allclose(solution.v, ref['v'], rtol=1e-3)
```

## Debugging Failed Tests

### Verbose Output

```bash
pytest tests/test_wind_model.py::test_specific -vv
```

### Debug with pdb

```python
def test_with_debugging():
    """Test with debugger."""
    import pdb
    
    model = WindModel(SFR=10.0)
    pdb.set_trace()  # Drops into debugger
    solution = model.run()
```

### Print Debugging

```python
def test_with_prints():
    """Test with print statements."""
    model = WindModel(SFR=10.0)
    
    # Use -s flag to see prints
    print(f"Model parameters: {model.get_parameters()}")
    solution = model.run()
    print(f"Terminal velocity: {solution.v_terminal}")
```

## Writing Good Tests

### 1. Clear Names

```python
# Good
def test_cooling_time_at_high_temperature():

# Bad  
def test_1():
```

### 2. Single Responsibility

```python
# Good - tests one thing
def test_sonic_point_regularization():
    # Only tests sonic point handling

# Bad - tests multiple things
def test_everything():
    # Tests model, cooling, and plotting
```

### 3. Good Assertions

```python
# Good - specific assertion
assert 450 < solution.v_terminal < 550  # km/s

# Bad - too vague
assert solution.v_terminal > 0
```

### 4. Document Expected Behavior

```python
def test_cloud_mass_conservation():
    """
    Test that cloud mass is conserved when no mass transfer.
    
    For f_turb0 = 0, clouds shouldn't exchange mass with hot phase.
    """
```

## See Also

- [Contributing Guide](contributing.md)
- [Code Review](code_review.md)
- [Bug Fixes](fixes.md)