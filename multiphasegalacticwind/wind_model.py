"""
Simple API wrapper for the multiphase galactic wind model.

This provides a clean interface while preserving all the original physics and units.
"""

import numpy as np
from scipy.integrate import solve_ivp
import os
from typing import Optional, Tuple, Callable, Union, Any, Dict

# Import necessary items from core physics
from .core_physics import (
    setup_cloud_powerlaw_distribution, Wind_Evo, Hot_Wind_Evo,
    create_supersonic_event, create_subsonic_event, create_wind_negative_event,
    create_cold_wind_event, create_all_clouds_frozen_event,
    create_cloud_density_low_event, create_cloud_velocity_low_event,
    create_progress_event, create_step_size_event
)
from .constants import *
from .config import WindConfig


class WindModel:
    """
    Simple wrapper for the multiphase galactic wind model.

    All units are preserved from the original code:
    - Masses in solar masses (Msun)
    - Velocities in km/s
    - Distances in kpc
    - Densities in cm^-3
    - Temperatures in K

    Metallicity Parameters
    ----------------------
    This model tracks two metallicities (both relative to solar):
    1. config.Z_hot_over_Z_solar: Hot gas metallicity in solar units (for cooling and initial wind)
    2. config.Z_cloud_over_Z_solar: Initial cloud metallicity in solar units
    """

    def __init__(self,
                 # Galaxy properties
                 v_circ: float = 150.0,         # km/s, circular velocity
                 redshift: float = 0.0,

                 # Wind launch properties
                 SFR: float = 20.0,             # Msun/yr, star formation rate
                 eta_M: float = 0.1,            # hot phase mass loading
                 eta_M_cold: Optional[float] = None,      # cold phase mass loading (also accepts eta_M_cold_tot)
                 eta_M_cold_tot: Optional[float] = None,  # alias for eta_M_cold
                 eta_E: float = 1.0,            # energy loading

                 # Sonic point properties
                 r_star_kpc: Optional[float] = None,      # kpc, sonic radius (also accepts r0_kpc)
                 r0_kpc: Optional[float] = None,          # alias for r_star_kpc

                 # Cloud properties
                 cloud_mass_range: Optional[Tuple[float, float]] = None,          # Msun, min and max cloud mass
                 log_M_cloud_min: Optional[float] = None,           # log10(M_min/Msun) - alternative to cloud_mass_range
                 log_M_cloud_max: Optional[float] = None,           # log10(M_max/Msun) - alternative to cloud_mass_range
                 cloud_alpha: float = 2.0,                # power law slope
                 N_cloud_species: int = 10,             # number of cloud mass bins
                 T_cl: Optional[float] = None,                      # K, cloud temperature (also accepts T_cloud)
                 T_cloud: Optional[float] = None,                   # alias for T_cl

                 # Solver settings
                 r_max_kpc: float = 100.0,      # kpc, maximum radius
                 rtol: float = 1e-8,
                 atol: float = 1e-10,

                 # Progress reporting
                 progress_callback: Optional[Union[str, Callable[[float, float, int], None]]] = None,  # Function(r_current, r_max, n_steps)
                 progress_interval: float = 10.0,  # kpc, interval for progress updates

                 # Configuration
                 config: Optional[WindConfig] = None,          # WindConfig instance
                 **config_kwargs: Any) -> None:
        """
        Initialize the wind model with galaxy and wind parameters.

        Parameters
        ----------
        v_circ : float, optional
            Circular velocity in km/s (default: 150.0)
        redshift : float, optional
            Redshift for cooling function (default: 0.0)
        SFR : float, optional
            Star formation rate in Msun/yr (default: 20.0)
        eta_M : float, optional
            Hot phase mass loading factor (default: 0.1)
        eta_M_cold, eta_M_cold_tot : float, optional
            Cold phase mass loading factor. Can use either name (default: 1.0)
        eta_E : float, optional
            Energy loading factor (default: 1.0)
        r_star_kpc, r0_kpc : float, optional
            Sonic radius in kpc. Can use either name (default: 0.3)
        cloud_mass_range : tuple, optional
            (min, max) cloud mass in Msun (default: (1, 1e5))
        log_M_cloud_min, log_M_cloud_max : float, optional
            Alternative to cloud_mass_range: log10(M/Msun) values
        cloud_alpha : float, optional
            Cloud mass distribution power law slope (default: 2.0)
        N_cloud_species : int, optional
            Number of cloud mass bins (default: 10)
        T_cl, T_cloud : float, optional
            Cloud temperature in K. Can use either name (default: 1e4)
        r_max_kpc : float, optional
            Maximum integration radius in kpc (default: 100.0)
        rtol, atol : float, optional
            Integration tolerances (default: 1e-8, 1e-10)
        progress_callback : callable or 'print', optional
            Function to report progress during integration. If 'print', uses
            built-in progress printer. Function signature: f(r_current, r_max, n_steps)
        progress_interval : float, optional
            Radius interval in kpc for progress updates (default: 10.0)
        config : WindConfig, optional
            Configuration object with model parameters. If None, uses defaults.
        **config_kwargs : dict
            Additional parameters to override in the configuration.

        Examples
        --------
        Use default configuration:
        >>> model = WindModel(SFR=10.0)

        Use legacy-style parameters:
        >>> model = WindModel(SFR=20.0, eta_M_cold_tot=0.0001, r0_kpc=0.3,
        ...                   log_M_cloud_min=1, log_M_cloud_max=5, T_cloud=1e4)

        Override specific config parameters:
        >>> model = WindModel(SFR=10.0, f_turb0=0.2, drag_coeff=0.3)
        """

        # Handle parameter aliases
        if eta_M_cold is None and eta_M_cold_tot is not None:
            eta_M_cold = eta_M_cold_tot
        elif eta_M_cold is None:
            eta_M_cold = 1.0  # default

        if r_star_kpc is None and r0_kpc is not None:
            r_star_kpc = r0_kpc
        elif r_star_kpc is None:
            r_star_kpc = 0.3  # default

        # Handle cloud mass range
        if cloud_mass_range is None:
            if log_M_cloud_min is not None and log_M_cloud_max is not None:
                cloud_mass_range = (10**log_M_cloud_min, 10**log_M_cloud_max)
            else:
                cloud_mass_range = (1, 1e5)  # default

        # Set up configuration
        if config is None:
            # Add redshift to config_kwargs if not already there
            if 'redshift' not in config_kwargs:
                config_kwargs['redshift'] = redshift
            # Add T_cl to config_kwargs if provided
            if T_cl is not None or T_cloud is not None:
                config_kwargs['T_cl'] = T_cl if T_cl is not None else T_cloud
            config = WindConfig(**config_kwargs)
        else:
            # Override any parameters passed as kwargs
            for key, value in config_kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            # Also set redshift on config if provided
            if redshift != config.redshift:
                config.redshift = redshift
            # Also set T_cl on config if provided
            if T_cl is not None or T_cloud is not None:
                config.T_cl = T_cl if T_cl is not None else T_cloud
        self.config = config

        # Store parameters
        self.v_circ = v_circ
        self.redshift = redshift
        self.SFR = SFR
        self.eta_M = eta_M
        self.eta_M_cold = eta_M_cold
        self.eta_E = eta_E
        self.r_star_kpc = r_star_kpc
        self.r_max_kpc = r_max_kpc
        self.rtol = rtol
        self.atol = atol
        self.progress_callback = progress_callback
        self.progress_interval = progress_interval

        # Warn about potentially problematic parameters
        if eta_M < 0.15 and eta_M_cold > 2 * eta_M:
            import warnings
            warnings.warn(
                f"Warning: eta_M={eta_M:.2f} and eta_M_cold={eta_M_cold:.2f} may lead to numerical issues.\n"
                "The cold mass loading is much higher than hot mass loading, which can cause:\n"
                "  - Excessive cooling leading to negative pressure/density\n"
                "  - Numerical stiffness causing integration to hang\n"
                "Consider using more balanced parameters (e.g., eta_M=0.2, eta_M_cold=0.2)",
                UserWarning,
                stacklevel=2
            )

        # Set up cloud distribution
        log_M_cloud_min = np.log10(cloud_mass_range[0])
        log_M_cloud_max = np.log10(cloud_mass_range[1])
        self.N_cloud_species = N_cloud_species

        # Use the original function to set up clouds
        self.M_cloud0, self.eta_M_cold_array, self.Mdot_cold0, self.Ndot_cloud0 = \
            setup_cloud_powerlaw_distribution(
                log_M_cloud_min, log_M_cloud_max, N_cloud_species,
                alpha_cloud=cloud_alpha, eta_M_cold_tot=eta_M_cold, SFR=SFR*Msun/yr
            )

        # Validate parameters
        self._validate_parameters()
        
        # Calculate sonic point conditions from physics
        self._calculate_sonic_point_conditions()

        # Load cooling table
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.cooling_table_path = os.path.join(data_dir, 'Lambda_tab_redshifts.npz')

    @property
    def T_cl(self) -> float:
        """Cloud temperature in K. For backward compatibility."""
        return self.config.T_cl
    
    def _validate_parameters(self) -> None:
        """
        Validate that all parameters are physically reasonable.
        
        Raises
        ------
        ValueError
            If any parameter is outside valid range
        """
        # Basic parameter checks
        if self.SFR <= 0:
            raise ValueError(f"SFR must be positive, got {self.SFR}")
        if self.v_circ <= 0:
            raise ValueError(f"v_circ must be positive, got {self.v_circ}")
        if self.eta_M <= 0:
            raise ValueError(f"eta_M must be positive, got {self.eta_M}")
        if self.eta_M_cold < 0:
            raise ValueError(f"eta_M_cold must be non-negative, got {self.eta_M_cold}")
        if self.eta_E <= 0:
            raise ValueError(f"eta_E must be positive, got {self.eta_E}")
        if self.r_star_kpc <= 0:
            raise ValueError(f"r_star_kpc must be positive, got {self.r_star_kpc}")
        if self.r_max_kpc <= self.r_star_kpc:
            raise ValueError(f"r_max_kpc must be greater than r_star_kpc, got {self.r_max_kpc} <= {self.r_star_kpc}")
        
        # Cloud mass range checks
        if self.M_cloud0[0] <= 0:
            raise ValueError(f"Minimum cloud mass must be positive")
        if self.N_cloud_species > 1 and self.M_cloud0[-1] <= self.M_cloud0[0]:
            raise ValueError(f"Maximum cloud mass must be greater than minimum")
        
        # Numerical tolerances
        if self.rtol <= 0 or self.rtol >= 1:
            raise ValueError(f"rtol must be in (0, 1), got {self.rtol}")
        if self.atol <= 0:
            raise ValueError(f"atol must be positive, got {self.atol}")
        
        # Config validation
        self.config.validate()

    def _calculate_sonic_point_conditions(self) -> None:
        """
        Calculate sonic point conditions from energy and mass injection rates.

        This follows the analytic solution for spherical winds at the sonic point
        where Mach = 1 + epsilon. The conditions are uniquely determined by the
        energy and mass injection rates.
        """
        # Convert to CGS units
        SFR_cgs = self.SFR * Msun / yr
        r_star = self.r_star_kpc * kpc

        # Mass and energy injection rates
        Mdot = self.eta_M * SFR_cgs  # g/s
        Edot = self.eta_E * (self.config.E_SN / (self.config.mstar * Msun)) * SFR_cgs  # erg/s

        # Sonic point Mach number
        Mach0 = 1.0 + self.config.sonic_point_offset
        gamma = 5.0/3.0  # Adiabatic index

        # Solve for sonic point velocity from energy conservation
        # v0 = sqrt(Edot/Mdot) * (1/((gamma-1)*Mach0) + 1/2)^(-1/2)
        v0 = np.sqrt(Edot/Mdot) * (1.0/((gamma-1)*Mach0) + 0.5)**(-0.5)

        # Density from mass flux conservation
        rho0 = Mdot / (self.config.Omwind * r_star**2 * v0)

        # Pressure from Mach number definition
        P0 = rho0 * v0**2 / (Mach0**2 * gamma)

        # Convert to convenient units
        self.v_star = v0 / 1e5  # km/s
        self.n_star = rho0 / (self.config.mu * mp)  # cm^-3
        self.T_star = P0 / (rho0 / (self.config.mu * mp)) / kb  # K

        # Store for later use
        self.rho_star = rho0
        self.P_star = P0

    def run(self) -> 'Solution':
        """
        Run the wind model and return a Solution object.
        """
        # Convert units to CGS as expected by the core physics
        r_star = self.r_star_kpc * kpc
        v_star_cgs = self.v_star * 1e5  # km/s to cm/s
        v_circ_cgs = self.v_circ * 1e5

        # Use pre-calculated sonic point values
        rho_star = self.rho_star
        P_star = self.P_star

        # Handle cloud_radial_offset if specified
        if self.config.cloud_radial_offset > 0:
            # First run hot-only solution to get conditions at offset radius
            r_start_offset = r_star * (1.0 + self.config.cloud_radial_offset)

            # Hot-only initial conditions
            y0_hot = np.array([v_star_cgs, rho_star, P_star])

            # Source term parameters for hot wind
            Edot = self.eta_E * (self.config.E_SN / (self.config.mstar * Msun)) * (self.SFR * Msun/yr)  # erg/s
            Mdot = self.eta_M * (self.SFR * Msun/yr)  # g/s
            r0 = r_star
            source_volume = 4./3. * np.pi * r0**3
            Edot_per_Vol = Edot / source_volume
            Mdot_per_Vol = Mdot / source_volume
            params_hot = (v_circ_cgs, True, r0, Edot_per_Vol, Mdot_per_Vol)

            # Integrate hot-only solution to offset radius
            sol_hot_offset = solve_ivp(
                lambda r, y: Hot_Wind_Evo(r, y, params_hot),
                [r_star, r_start_offset], y0_hot,
                rtol=self.rtol, atol=self.atol,
                dense_output=True
            )

            # Extract hot gas conditions at offset radius
            v_offset = sol_hot_offset.y[0, -1]
            rho_offset = sol_hot_offset.y[1, -1]
            P_offset = sol_hot_offset.y[2, -1]

            # Set up initial conditions at offset radius
            y0 = np.zeros(4 + 3*self.N_cloud_species)
            y0[0] = v_offset
            y0[1] = rho_offset
            y0[2] = P_offset
            y0[3] = rho_offset * self.config.Z_hot_over_Z_solar * Z_solar  # rhoZ_wind
            y0[4:4+self.N_cloud_species] = self.M_cloud0  # Already in grams
            y0[4+self.N_cloud_species:4+2*self.N_cloud_species] = self.config.v_cloud_init * 1e5  # v_cloud array in cm/s
            y0[4+2*self.N_cloud_species:] = self.config.Z_cloud_over_Z_solar * Z_solar  # Z_cloud array (absolute)

            # Integration span from offset radius
            r_span = [r_start_offset, self.r_max_kpc * kpc]

        else:
            # Standard initial conditions at sonic point
            y0 = np.zeros(4 + 3*self.N_cloud_species)
            y0[0] = v_star_cgs
            y0[1] = rho_star
            y0[2] = P_star
            y0[3] = rho_star * self.config.Z_hot_over_Z_solar * Z_solar  # rhoZ_wind
            y0[4:4+self.N_cloud_species] = self.M_cloud0  # Already in grams
            y0[4+self.N_cloud_species:4+2*self.N_cloud_species] = self.config.v_cloud_init * 1e5  # v_cloud array in cm/s
            y0[4+2*self.N_cloud_species:] = self.config.Z_cloud_over_Z_solar * Z_solar  # Z_cloud array (absolute)

            # Integration span from sonic point
            r_span = [r_star, self.r_max_kpc * kpc]

        # Calculate source term parameters
        # Energy and mass injection rates
        Edot = self.eta_E * (self.config.E_SN / (self.config.mstar * Msun)) * (self.SFR * Msun/yr)  # erg/s
        Mdot = self.eta_M * (self.SFR * Msun/yr)  # g/s

        # Source volume (sphere of radius r_star)
        r0 = r_star  # injection radius same as sonic radius
        source_volume = 4./3. * np.pi * r0**3

        # Volume-averaged source terms
        Edot_per_Vol = Edot / source_volume  # erg/s/cm^3
        Mdot_per_Vol = Mdot / source_volume  # g/s/cm^3

        # Parameters for Wind_Evo
        # Calculate injection radius as fraction of r0
        injection_radius = self.config.cold_cloud_injection_radial_extent_frac * r0
        injection_power = self.config.cold_cloud_injection_radial_power
        config_dict = self.config.to_dict()

        # Pre-calculate cooling callable for efficiency (backend-selectable).
        if self.config.cooling_backend == 'topaz':
            from .topaz_cooling import get_lambda_p_rho_callable_topaz, load_cooling_table
            topaz_table = load_cooling_table(self.config.topaz_cooling_table_path)
            cooling_interpolator = get_lambda_p_rho_callable_topaz(
                self.config.mu,
                self.config.Z_hot_over_Z_solar,
                table_path=self.config.topaz_cooling_table_path,
            )
            # Reuse already-loaded table in Wind_Evo for mixed-layer tcool calls.
            config_dict['_topaz_table'] = topaz_table
        elif self.config.cooling_backend == 'legacy':
            from .cooling import get_cooling_interpolator
            cooling_interpolator = get_cooling_interpolator(
                self.config.mu, self.config.Z_hot_over_Z_solar, self.config.redshift
            )
        else:
            raise ValueError(
                f"Unknown cooling backend: {self.config.cooling_backend}. "
                "Expected 'legacy' or 'topaz'."
            )

        # Extended params tuple including source terms and cooling interpolator
        params = (v_circ_cgs, self.Ndot_cloud0, self.config.T_cl,
                  injection_radius, injection_power, config_dict,
                  r0, Edot_per_Vol, Mdot_per_Vol, cooling_interpolator)

        # Check initial Mach number to determine which events to include
        # Calculate initial Mach number
        cs_sq_initial = gamma * y0[2] / y0[1]  # gamma * P / rho
        mach_initial = y0[0] / np.sqrt(cs_sq_initial)

        # Create event functions with params
        from .core_physics import (create_negative_pressure_event,
                                   create_negative_density_event,
                                   create_nan_state_event)

        wind_negative = create_wind_negative_event(params)
        cold_wind = create_cold_wind_event(params)
        all_clouds_frozen = create_all_clouds_frozen_event(params)
        cloud_density_low = create_cloud_density_low_event(params)
        cloud_velocity_low = create_cloud_velocity_low_event(params)
        negative_pressure = create_negative_pressure_event(params)
        negative_density = create_negative_density_event(params)
        nan_state = create_nan_state_event(params)
        step_size = create_step_size_event(params, min_relative_step=1e-7, n_small_steps=20)

        # Create list of events - always include these
        events = [wind_negative,
                  cold_wind,
                  all_clouds_frozen,
                  cloud_density_low,
                  cloud_velocity_low,
                  negative_pressure,
                  negative_density,
                  nan_state,
                  step_size]

        # Only add supersonic event if starting subsonic
        # (Don't need it if already supersonic)
        if mach_initial < 1.0:
            supersonic = create_supersonic_event(params)
            events.insert(0, supersonic)  # Add at beginning for consistency

        # Add subsonic event for all runs starting supersonic
        if mach_initial > 1.0:
            subsonic = create_subsonic_event(params)
            events.insert(0, subsonic)  # Add at beginning for consistency

        # Add progress event if callback provided
        if self.progress_callback is not None:
            # Handle special case for 'print' callback
            if self.progress_callback == 'print':
                import sys
                def print_progress(r_current, r_max, n_steps):
                    percent = 100 * r_current / r_max
                    print(f"\rProgress: {percent:5.1f}% (r = {r_current/kpc:6.1f} kpc, steps = {n_steps})",
                          end='', flush=True)
                    if r_current >= r_max * 0.99:  # Near completion
                        print()  # New line at end
                progress_cb = print_progress
            else:
                progress_cb = self.progress_callback

            progress_event = create_progress_event(
                params, r_span[0], self.progress_interval * kpc, progress_cb, r_span[1]
            )
            events.append(progress_event)

        # Run the integration with maximum evaluations to prevent hanging
        try:
            solve_kwargs = {
                'rtol': self.rtol,
                'atol': self.atol,
                'dense_output': True,
                'events': events,
            }
            if self.config.solver_max_step_kpc is not None:
                solve_kwargs['max_step'] = self.config.solver_max_step_kpc * kpc
            if self.config.solver_first_step_kpc is not None:
                solve_kwargs['first_step'] = self.config.solver_first_step_kpc * kpc

            sol = solve_ivp(
                lambda r, y: Wind_Evo(r, y, params),
                r_span, y0,
                **solve_kwargs,
            )
        except ValueError as e:
            if "`ts` must be strictly increasing or decreasing" in str(e):
                # Integration got stuck - return a terminated solution
                print(f"\nWarning: Integration terminated early due to numerical issues at r ≈ {r_span[0]/kpc:.2f} kpc")
                print(f"  This typically happens with eta_M={self.eta_M:.2f} and eta_M_cold={self.eta_M_cold:.2f}")
                print(f"  The solution becomes unphysical and cannot continue.")

                # Create a minimal failed solution
                # Use a simple object that has the required attributes
                class FailedSolution:
                    def __init__(self, r0, y0, eta_M, eta_M_cold):
                        self.t = np.array([r0])
                        self.y = y0.reshape(-1, 1)
                        self.status = -1
                        self.message = f"Integration failed: eta_M={eta_M}, eta_M_cold={eta_M_cold} causes numerical stiffness"
                        self.success = False
                        self.t_events = []
                        self.y_events = []
                        self.nfev = 0
                        self.njev = 0
                        self.nlu = 0

                    def __call__(self, t):
                        # Return initial conditions for any query
                        return self.y[:, 0]

                sol = FailedSolution(r_span[0], y0, self.eta_M, self.eta_M_cold)
            else:
                raise  # Re-raise if it's a different error

        # Also run hot-only solution for comparison
        y0_hot = y0[:3].copy()
        # Include source terms for hot wind
        params_hot = (v_circ_cgs, True, r0, Edot_per_Vol, Mdot_per_Vol)

        sol_hot = solve_ivp(
            lambda r, y: Hot_Wind_Evo(r, y, params_hot),
            r_span, y0_hot,
            rtol=self.rtol, atol=self.atol,
            dense_output=True,
            events=[create_wind_negative_event(params_hot)]
        )

        return Solution(sol, sol_hot, self)


class Solution:
    """
    Container for wind model solution with convenient access to results.
    """

    def __init__(self, sol: Any, sol_hot: Any, model: WindModel) -> None:
        self.sol = sol
        self.sol_hot = sol_hot
        self.model = model

        # Extract solution arrays
        self.r = sol.t / kpc  # Convert to kpc
        self.v = sol.y[0] / 1e5  # Convert to km/s
        self.rho = sol.y[1]
        self.P = sol.y[2]
        self.rhoZ = sol.y[3]
        self.Z = self.rhoZ / self.rho  # metallicity
        self.M_clouds = sol.y[4:4+model.N_cloud_species] / Msun  # Convert to Msun
        self.v_cl = sol.y[4+model.N_cloud_species:4+2*model.N_cloud_species] / 1e5  # km/s
        self.Z_cl = sol.y[4+2*model.N_cloud_species:]  # absolute metallicity

        # Hot-only solution
        self.r_hot = sol_hot.t / kpc
        self.v_hot = sol_hot.y[0] / 1e5
        self.rho_hot = sol_hot.y[1]
        self.P_hot = sol_hot.y[2]

        # Derived quantities
        mu = model.config.mu
        self.n = self.rho / (mu * mp)  # number density
        self.T = self.P / (self.rho / (mu * mp)) / kb  # temperature
        self.n_hot = self.rho_hot / (mu * mp)
        self.T_hot = self.P_hot / (self.rho_hot / (mu * mp)) / kb

        # Mass fluxes
        solid_angle = model.config.Omwind
        self.Mdot = solid_angle * sol.t**2 * self.rho * sol.y[0] / (Msun/yr)
        self.Mdot_hot = solid_angle * sol_hot.t**2 * self.rho_hot * sol_hot.y[0] / (Msun/yr)

        # Total cloud mass
        self.M_cloud_tot = np.sum(self.M_clouds, axis=0)

    def interpolate(self, r_eval: Union[float, np.ndarray]) -> np.ndarray:
        """
        Interpolate solution at specific radii (in kpc).
        """
        r_eval_cgs = np.atleast_1d(r_eval) * kpc
        return self.sol.sol(r_eval_cgs)

    @property
    def v_at_10kpc(self) -> float:
        """Velocity at 10 kpc in km/s."""
        if self.r[-1] >= 10:
            return np.interp(10, self.r, self.v)
        else:
            return np.nan

    @property
    def mass_loading_at_10kpc(self) -> float:
        """Mass loading factor at 10 kpc."""
        if self.r[-1] >= 10:
            Mdot_10kpc = np.interp(10, self.r, self.Mdot)
            return Mdot_10kpc / self.model.SFR
        else:
            return np.nan

    def calculate_velocity_moments(self, **kwargs: Any) -> Dict[str, float]:
        """
        Calculate velocity moments from column density distribution.

        This uses the column density distribution (observable quantity)
        rather than number density distribution.

        Parameters
        ----------
        **kwargs : dict
            Arguments for column density distribution calculation

        Returns
        -------
        moments : dict
            Dictionary with mean, dispersion, etc.
        """
        from .observables import calculate_velocity_moments
        v_cloud, dN_dv = self.calculate_column_density_distribution(**kwargs)
        return calculate_velocity_moments(v_cloud, dN_dv)

    def calculate_column_density_distribution(self, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate column density distribution dN/dv in cm^-2 / (km/s).

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to observables.calculate_column_density_distribution

        Returns
        -------
        v_cloud : array
            Cloud velocities [km/s]
        dN_dv_column : array
            Column density distribution [cm^-2 / (km/s)]
        """
        from .observables import calculate_column_density_distribution
        return calculate_column_density_distribution(self, **kwargs)

    def calculate_column_density_by_species(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculate column density distribution for each cloud species.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to observables.calculate_column_density_by_species

        Returns
        -------
        v_cloud : array
            Cloud velocities [km/s]
        dN_dv_species : dict
            Dictionary with 'total', 'species' list, and 'M_cloud0'
        """
        from .observables import calculate_column_density_by_species
        return calculate_column_density_by_species(self, **kwargs)
