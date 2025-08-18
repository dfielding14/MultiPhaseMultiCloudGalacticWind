# Plotting Module - Comprehensive Code Review

## Module Overview
**File**: `multiphasegalacticwind/plotting.py`  
**Lines**: 457  
**Purpose**: Publication-quality visualization functions for wind model results

## Dependencies (Lines 1-18)

### External Libraries
- `matplotlib`: Core plotting
- `cmasher`: Scientific colormaps (optional)
- `mpl_toolkits.axes_grid1`: Inset axes for colorbars

**Good Practice**: Optional cmasher with fallback to viridis

## Function Analysis

### 1. `setup_plotting_style()` (Lines 20-64)
**Purpose**: Configure matplotlib for publication-quality plots
**Key Settings**:
- DPI: 200 (high resolution)
- Ticks: Inward on all sides with minors visible
- LaTeX: Optional rendering with cmbright font

**Issues**:
- Line 62: Bare except for LaTeX setup
- No way to revert to default style
- Global state modification

### 2. `plot_wind_solution()` (Lines 66-222)
**Purpose**: Create standard 3-panel wind solution plot
**Panels**:
1. Velocity (wind, clouds, sound speed)
2. Mass flux (wind, clouds)
3. Cloud masses

**Key Features**:
- Multi-species cloud visualization with color coding
- Hot-only comparison
- Embedded colorbar for cloud masses

**Implementation Details** (Lines 90-95):
```python
if HAS_CMASHER and show_clouds:
    cloud_colors = cmr.take_cmap_colors('cmr.guppy', N_species, ...)
else:
    cloud_colors = plt.cm.viridis(np.linspace(0, 1, N_species))
```

**Cloud Mass Flux Calculation** (Lines 133-151):
```python
# Calculate injection function
injection_function = np.where(r < r_inj, (r/r_inj)**p, 1.0)
# Cloud mass flux
Mdot_cl = Ndot_cloud * M_cloud * injection_function
```

**Issues**:
- Line 102-103: Complex masking logic repeated multiple times
- Line 130: Uses `kpc` from constants but defined locally
- Line 144: Complex unit conversion inline
- Line 252: References undefined `v_cl` attribute

### 3. `plot_profiles()` (Lines 225-284)
**Purpose**: Plot various wind profile quantities
**Supported Quantities**:
- velocity
- density
- temperature
- pressure
- mass_flux
- metallicity

**Good Design**:
- Flexible quantity selection
- Shared x-axis for multi-panel
- Clean separation of plotting logic

**Issues**:
- Line 252: `solution.v_cl` undefined - should be `solution.v_cl[0]` or mean
- Line 273: `solution.Z_cl` may need indexing for multi-species

### 4. `plot_column_density_distribution()` (Lines 288-457)
**Purpose**: Plot dN/dv column density distribution
**Features**:
- Individual species breakdown
- Statistical moments overlay
- Embedded cloud mass colorbar

**Complex Logic** (Lines 330-395):
Handles two modes:
1. Show all species with colors
2. Show single species or total

**Colorbar Implementation** (Lines 354-391):
- Creates inset axes
- Maps colors to cloud masses
- Smart labeling for many species

**Moments Display** (Lines 410-436):
```python
if show_moments:
    moments = calculate_velocity_moments(v_cloud, dN_dv)
    # Display mean and dispersion
    ax.axvline(mean_v, ls='--')
    ax.axvline(mean_v ± disp_v, ls=':')
```

**Issues**:
- Line 345-348: Loop could be vectorized
- Line 363: Assumes `M_cloud0` in dict
- Duplicate colorbar code from `plot_wind_solution()`

## Design Patterns

### Color Management
- Uses cmasher 'guppy' colormap for clouds
- Falls back to viridis if unavailable
- Consistent color mapping across plots

### Layout Management
- `constrained_layout=True` for automatic spacing
- GridSpec for complex layouts
- Inset axes for colorbars

### Style Consistency
- All plots call `setup_plotting_style()`
- Consistent line styles (solid, dashed, dotted)
- Standard label formatting with LaTeX

## Critical Issues

### 1. **Undefined Attributes**
Lines 252, 273: References to `solution.v_cl` and `solution.Z_cl` that don't exist as simple attributes

### 2. **Repeated Code**
Colorbar implementation duplicated between functions (lines 179-216 and 354-391)

### 3. **Unit Handling**
Multiple inline unit conversions without clear documentation:
- Line 110: `c_s` calculation with hardcoded gamma
- Line 144: Complex Msun/yr conversion

### 4. **Masking Logic**
Repeated pattern:
```python
masked = np.ma.masked_where(M_clouds < M_cloud_min, data)
```
Should be extracted to utility function

## Documentation Needs

### High Priority
1. Document expected Solution object structure
2. Explain colorbar mapping logic
3. Document unit assumptions

### Medium Priority
1. Add examples for each plot type
2. Document style choices
3. Explain panel layout decisions

## Recommended Improvements

### Immediate Fixes

1. **Fix attribute references**:
```python
# Line 252 - handle cloud velocities properly
if hasattr(solution, 'v_cl') and len(solution.v_cl.shape) > 1:
    v_cl_mean = np.mean(solution.v_cl, axis=0)
else:
    v_cl_mean = solution.v_cl if hasattr(solution, 'v_cl') else None
```

2. **Extract colorbar function**:
```python
def add_cloud_mass_colorbar(ax, N_species, M_cloud0, location='lower left'):
    """Add standardized cloud mass colorbar."""
    # Unified implementation
```

3. **Create masking utility**:
```python
def mask_destroyed_clouds(data, M_clouds, M_cloud_min):
    """Mask data where clouds are destroyed."""
    return np.ma.masked_where(M_clouds < M_cloud_min, data)
```

### Future Enhancements

1. **Style manager**:
```python
class PlotStyle:
    def __init__(self, style='publication'):
        self.style = style
    
    def __enter__(self):
        self.old_params = mpl.rcParams.copy()
        self.apply_style()
    
    def __exit__(self, *args):
        mpl.rcParams.update(self.old_params)
```

2. **Plot builder pattern**:
```python
class WindPlotBuilder:
    def add_velocity_panel(self):
        # Add velocity plot
    def add_mass_flux_panel(self):
        # Add mass flux plot
    def build(self):
        return self.fig
```

3. **Animation support**:
```python
def animate_solution(solutions, parameter_name):
    """Animate solution evolution with parameter."""
```

## Performance Analysis

### Bottlenecks
1. Repeated masking operations
2. Loop over cloud species (not vectorized)
3. Multiple interpolation calls

### Optimization Opportunities
1. Vectorize species plotting
2. Cache color calculations
3. Pre-compute masks once

## Testing Requirements

### Visual Tests
1. Compare with reference plots
2. Test with edge cases (1 species, 100 species)
3. Verify colorbar labels

### Unit Tests
1. `test_color_generation`: Verify color mapping
2. `test_masking`: Check cloud destruction masking
3. `test_units`: Verify unit conversions

### Integration Tests
1. Test with actual Solution objects
2. Test all plot types
3. Test style consistency

## Code Quality Metrics

- **Documentation**: ~25% (minimal docstrings)
- **Type Hints**: 0% (none present)
- **Error Handling**: Poor (bare excepts)
- **Code Duplication**: Moderate (colorbar code)
- **Complexity**: Medium (longest function 156 lines)

## Summary

The plotting module provides publication-quality visualizations but needs:

1. **Bug fixes**: Undefined attribute references
2. **Refactoring**: Extract repeated code (colorbar, masking)
3. **Documentation**: Add examples and explain design choices
4. **Type hints**: Add for better IDE support
5. **Error handling**: Remove bare except clauses

The module successfully creates professional plots with:
- Consistent styling
- Multi-species visualization
- Flexible plot types
- Publication-ready output

Key strength is the aesthetic quality matching published papers.