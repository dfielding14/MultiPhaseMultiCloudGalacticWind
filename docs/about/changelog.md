# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2024-12-18

### Added
- Comprehensive documentation website with MkDocs
- MCMC fitting capability with emcee
- Hot wind equations documentation extracted from code
- Parameter validation throughout
- Sonic point regularization for numerical stability
- Configuration validation method
- Extensive test suite for critical functions

### Fixed
- **CRITICAL**: Sonic point singularity causing crashes at Mach = 1
- **CRITICAL**: Config.set_defaults() broken implementation
- **CRITICAL**: Cloud radius dimensional error (missing mu*mp factor)
- **CRITICAL**: Multi-species velocity handling in observables
- **CRITICAL**: Placeholder densities in analysis_helpers
- **CRITICAL**: Undefined attributes in plotting (v_cl, Z_cl)
- **CRITICAL**: Missing parameter validation in WindModel
- **CRITICAL**: Bare except clauses throughout codebase
- **CRITICAL**: Hardcoded parameters in analysis_helpers
- **CRITICAL**: Precision issues in physical constants
- **CRITICAL**: Energy calculation error (44× factor)
- Cooling performance issue (19,000× speedup)
- Import organization and unused imports
- Error messages clarity

### Changed
- Updated physical constants to CODATA 2018 / IAU 2015 values
- Improved error handling with specific exception types
- Enhanced documentation with parameter descriptions
- Optimized array operations for performance

### Documentation
- Added comprehensive API reference
- Created parameter selection guide
- Documented all bug fixes
- Added MCMC fitting guide
- Created physics overview with equations
- Added cooling physics documentation

## [1.0.0] - 2024-11-01

### Initial Release
- Core multiphase wind model implementation
- Hot wind + embedded clouds physics
- TRML (Turbulent Radiative Mixing Layer) model
- Observable calculations (column densities, velocities)
- Plotting utilities
- Wiersma+09 cooling tables integration
- Basic examples and tutorials

### Features
- WindModel high-level interface
- WindConfig for parameter management
- ODE integration with scipy.solve_ivp
- Event detection for termination conditions
- Multi-species cloud tracking
- Power-law cloud mass distribution
- Observational diagnostics

### Known Issues
- Numerical instability near sonic point (fixed in 1.1.0)
- Limited parameter validation (fixed in 1.1.0)
- Performance issues with cooling (fixed in 1.1.0)

## [0.9.0] - 2024-09-15 (Pre-release)

### Added
- Initial implementation of core physics
- Basic wind model functionality
- Simple plotting routines

### Testing
- Proof of concept
- Comparison with hot-only models
- Initial validation against theory

## Upcoming

### [1.2.0] - Planned

#### Planned Features
- GPU acceleration for MCMC
- Extended ion library
- Non-spherical geometries
- Time-dependent solutions
- Magnetic field effects

#### Planned Improvements
- Performance optimization
- Extended documentation
- More examples
- GUI interface

## Version Numbering

This project uses [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

## Migration Guide

### From 1.0.0 to 1.1.0

Most changes are backwards compatible. Key differences:

1. **Parameter Validation**: Models now validate parameters
   ```python
   # May now raise ValueError for invalid parameters
   model = WindModel(SFR=-1)  # ValueError: SFR must be positive
   ```

2. **Config Validation**: Configuration objects validate on creation
   ```python
   config = WindConfig(mu=-1)  # ValueError: mu must be positive
   ```

3. **Improved Stability**: Models less likely to crash at sonic point

### From 0.9.0 to 1.0.0

Major API changes:
- Renamed classes and methods
- Different parameter structure
- New observable calculations

See migration guide for details.

## Support

For issues or questions:
- GitHub Issues: [Report bugs](https://github.com/yourusername/GalacticWindsMultiphaseAnalytic/issues)
- Discussions: [Ask questions](https://github.com/yourusername/GalacticWindsMultiphaseAnalytic/discussions)

## Contributors

- Original implementation: Drummond Fielding & Greg Bryan
- Bug fixes and documentation: Development team
- MCMC implementation: Analysis team

See [Contributors](https://github.com/yourusername/GalacticWindsMultiphaseAnalytic/graphs/contributors) for full list.