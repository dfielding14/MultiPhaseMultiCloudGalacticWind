# Changelog

All notable changes to the MultiPhase MultiCloud Galactic Wind Model will be documented in this file.

## [Unreleased]

### Added
- New `analysis_helpers.py` module containing physical analysis functions
- Factory pattern for all event detection functions
- `sonic_point_tolerance` parameter in WindConfig for event detection
- Conditional event activation based on initial conditions
- Progress reporting functionality for long integrations
- Type hints throughout the codebase
- Comprehensive unit tests for cooling and config modules
- Parameter aliases for backward compatibility
- Cloud velocity and metallicity configuration options

### Changed
- Moved helper functions from `core_physics.py` to `analysis_helpers.py`
- Refactored event detection system to use consistent factory pattern
- Renamed `epsilon` to `sonic_point_tolerance` for clarity
- Simplified metallicity parameters to `Z_hot_over_Z_solar` and `Z_cloud_over_Z_solar`
- Supersonic event only active when starting subsonic
- Subsonic event only active when starting supersonic
- Updated README with event documentation

### Fixed
- Cloud metallicity units (now properly uses solar units)
- Parameter name mismatches causing integration failures
- Minimum cloud velocity safeguard to prevent numerical singularities
- Cloud mass units bug (removed double Msun multiplication)
- Integration failures from cloud velocity approaching zero

### Removed
- Duplicate event definitions
- Legacy `cloud_stop` function
- Global `epsilon` variable
- All development and debugging scripts

## [0.1.0] - Initial Release

- Initial public release of the multiphase galactic wind model
- Based on Fielding & Bryan (2024) paper
- Core physics implementation with ODE integration
- Velocity and column density observables
- Matplotlib-based plotting functions
- Comprehensive documentation and examples