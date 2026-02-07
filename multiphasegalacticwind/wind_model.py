"""
Simple API wrapper for the multiphase galactic wind model.

This provides a clean interface while preserving all the original physics and units.
"""

import os
from typing import Optional, Tuple, Callable, Union, Any, Dict

import jax
import numpy as np

# Enable physically important 64-bit precision for stable ODE integration.
jax.config.update("jax_enable_x64", True)

from .core_physics import setup_cloud_powerlaw_distribution
from .jax_physics import (
    build_jax_wind_params,
    integrate_hot_wind_rk4_scan,
    integrate_wind_rk4_scan,
    jacobian_wind_rhs_state,
)
from .constants import *
from .config import WindConfig
from .topaz_cooling import load_cooling_table


class _LinearDenseOutput:
    """Lightweight linear dense-output replacement for post-integration queries."""

    def __init__(self, t: np.ndarray, y: np.ndarray) -> None:
        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def __call__(self, t_eval: Union[float, np.ndarray]) -> np.ndarray:
        t_query = np.atleast_1d(np.asarray(t_eval, dtype=float))
        out = np.vstack([np.interp(t_query, self.t, self.y[i]) for i in range(self.y.shape[0])])
        if np.ndim(t_eval) == 0:
            return out[:, 0]
        return out


class _OdeResult:
    """Minimal solution container mirroring scipy's OdeResult attributes used downstream."""

    def __init__(self, t: np.ndarray, y: np.ndarray, success: bool = True, message: str = "") -> None:
        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.status = 0 if success else -1
        self.message = message
        self.success = success
        self.t_events = []
        self.y_events = []
        self.nfev = 0
        self.njev = 0
        self.nlu = 0
        self.sol = _LinearDenseOutput(self.t, self.y)


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
        if self.config.cooling_backend != "topaz":
            raise ValueError(
                "JAX migration currently supports only cooling_backend='topaz'. "
                f"Got {self.config.cooling_backend!r}."
            )

        r_star = self.r_star_kpc * kpc
        v_star_cgs = self.v_star * 1e5
        v_circ_cgs = self.v_circ * 1e5
        rho_star = self.rho_star
        P_star = self.P_star

        # Energy and mass injection rates.
        Edot = self.eta_E * (self.config.E_SN / (self.config.mstar * Msun)) * (self.SFR * Msun / yr)
        Mdot = self.eta_M * (self.SFR * Msun / yr)
        r0 = r_star
        source_volume = 4.0 / 3.0 * np.pi * r0**3
        Edot_per_Vol = Edot / source_volume
        Mdot_per_Vol = Mdot / source_volume

        # Initial hot state.
        y0_hot = np.array([v_star_cgs, rho_star, P_star], dtype=float)

        # Use a conservative fixed RK4 cap to maintain moment-level agreement.
        # solver_max_step_kpc remains an upper bound, and solver_first_step_kpc
        # seeds a geometric launch ramp for sonic-point stability.
        effective_step_kpc = min(self.config.solver_max_step_kpc or 0.3, 0.02)
        first_step_kpc = min(self.config.solver_first_step_kpc or effective_step_kpc, effective_step_kpc)

        def build_r_grid(r_start_val: float, r_end_val: float) -> np.ndarray:
            points = [float(r_start_val)]
            h = max(first_step_kpc * kpc, 1e-12 * kpc)
            h_cap = max(effective_step_kpc * kpc, h)

            # Geometric warm-up near the launch radius to avoid sonic-point blowups.
            while h < h_cap and points[-1] + h < r_end_val:
                points.append(points[-1] + h)
                h = min(h * 2.0, h_cap)

            current = points[-1]
            if current >= r_end_val:
                return np.asarray(points, dtype=float)

            n_uniform = max(1, int(np.ceil((r_end_val - current) / h_cap)))
            tail = np.linspace(current, r_end_val, n_uniform + 1, dtype=float)[1:]
            return np.concatenate([np.asarray(points, dtype=float), tail])

        # Handle cloud radial offset by integrating hot-only to the offset radius.
        if self.config.cloud_radial_offset > 0:
            r_start_offset = r_star * (1.0 + self.config.cloud_radial_offset)
            r_offset_grid = build_r_grid(r_star, r_start_offset)
            hot_offset_track = np.asarray(
                integrate_hot_wind_rk4_scan(
                    r_offset_grid,
                    y0_hot,
                    v_circ_cgs,
                    True,
                    r0,
                    Edot_per_Vol,
                    Mdot_per_Vol,
                )
            )
            v_offset, rho_offset, P_offset = hot_offset_track[-1]
            r_start = r_start_offset
        else:
            v_offset, rho_offset, P_offset = y0_hot
            r_start = r_star

        # Build full initial state.
        y0 = np.zeros(4 + 3 * self.N_cloud_species, dtype=float)
        y0[0] = v_offset
        y0[1] = rho_offset
        y0[2] = P_offset
        y0[3] = rho_offset * self.config.Z_hot_over_Z_solar * Z_solar
        y0[4 : 4 + self.N_cloud_species] = self.M_cloud0
        y0[4 + self.N_cloud_species : 4 + 2 * self.N_cloud_species] = self.config.v_cloud_init * 1e5
        y0[4 + 2 * self.N_cloud_species :] = self.config.Z_cloud_over_Z_solar * Z_solar

        r_end = self.r_max_kpc * kpc
        r_grid = build_r_grid(r_start, r_end)

        config_dict = self.config.to_dict()
        topaz_table = load_cooling_table(self.config.topaz_cooling_table_path)
        config_dict["_topaz_table"] = topaz_table

        injection_radius = self.config.cold_cloud_injection_radial_extent_frac * r0
        injection_power = self.config.cold_cloud_injection_radial_power

        jax_params = build_jax_wind_params(
            v_circ=v_circ_cgs,
            Ndot_cloud0=self.Ndot_cloud0,
            T_cloud=self.config.T_cl,
            injection_radius=injection_radius,
            injection_power=injection_power,
            config_dict=config_dict,
            r0=r0,
            Edot_per_Vol=Edot_per_Vol,
            Mdot_per_Vol=Mdot_per_Vol,
            table_path=self.config.topaz_cooling_table_path,
        )
        self._last_jax_params = jax_params
        self._last_r_grid = r_grid.copy()
        self._last_y0 = y0.copy()

        y_track = np.asarray(integrate_wind_rk4_scan(r_grid, y0, jax_params))
        finite = np.all(np.isfinite(y_track), axis=1)
        physical = (y_track[:, 0] > 0.0) & (y_track[:, 1] > 0.0) & (y_track[:, 2] > 0.0)
        valid = finite & physical
        invalid_idx = np.where(~valid)[0]
        if invalid_idx.size > 0:
            stop = max(2, int(invalid_idx[0]))
            success = False
            message = f"JAX integration terminated at r={r_grid[stop-1]/kpc:.2f} kpc due to unphysical state."
        else:
            stop = len(r_grid)
            success = True
            message = ""

        t_used = r_grid[:stop]
        y_used = y_track[:stop].T
        sol = _OdeResult(t_used, y_used, success=success, message=message)

        y0_hot_local = y0[:3].copy()
        hot_track = np.asarray(
            integrate_hot_wind_rk4_scan(
                t_used,
                y0_hot_local,
                v_circ_cgs,
                True,
                r0,
                Edot_per_Vol,
                Mdot_per_Vol,
            )
        )
        sol_hot = _OdeResult(t_used, hot_track.T, success=True, message="")

        if self.progress_callback == "print":
            print(
                f"Progress: 100.0% (r = {t_used[-1]/kpc:6.1f} kpc, steps = {len(t_used):d})"
            )
        elif callable(self.progress_callback):
            self.progress_callback(float(t_used[-1]), float(r_end), int(len(t_used)))

        return Solution(sol, sol_hot, self)

    def jacobian_rhs(self, r_kpc: float, state: np.ndarray) -> np.ndarray:
        """
        Return the Jacobian ∂(dstate/dr)/∂state evaluated at a radius/state.

        Parameters
        ----------
        r_kpc : float
            Radius in kpc.
        state : array
            Full multiphase state vector in internal CGS units.
        """
        if not hasattr(self, "_last_jax_params"):
            raise RuntimeError("Run the model once before requesting Jacobians.")
        jac = jacobian_wind_rhs_state(r_kpc * kpc, np.asarray(state, dtype=float), self._last_jax_params)
        return np.asarray(jac)

    def run_differentiable(self):
        """
        Return the last-run trajectory as a JAX array for autodiff workflows.

        The returned array has shape ``(N_r, N_state)`` and is generated by the
        JAX RK4 integrator directly.
        """
        if not hasattr(self, "_last_jax_params") or not hasattr(self, "_last_r_grid") or not hasattr(self, "_last_y0"):
            raise RuntimeError("Run the model once before requesting differentiable trajectories.")
        return integrate_wind_rk4_scan(
            self._last_r_grid,
            self._last_y0,
            self._last_jax_params,
        )


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
