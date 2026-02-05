"""
Plotting functions that preserve the publication-quality aesthetic from the original notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .constants import Msun, yr, kpc
from .plotting_helpers import (get_cloud_colors, add_cloud_mass_colorbar, 
                              mask_cloud_data)

# Try to import cmasher for colormaps
try:
    import cmasher as cmr
    HAS_CMASHER = True
except ImportError:
    HAS_CMASHER = False
    print("Warning: cmasher not installed. Using matplotlib colormaps instead.")


def setup_plotting_style():
    """
    Set up the publication-quality plotting style used in the papers.
    """
    # High DPI for publication quality
    matplotlib.rcParams['figure.dpi'] = 200

    # Ticks point inward on all sides
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True

    # Minor ticks visible
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['ytick.minor.visible'] = True

    # Tick dimensions
    matplotlib.rcParams['xtick.major.size'] = 2.75
    matplotlib.rcParams['ytick.major.size'] = 2.75
    matplotlib.rcParams['xtick.minor.size'] = 1.75
    matplotlib.rcParams['ytick.minor.size'] = 1.75

    # Tick line widths
    matplotlib.rcParams['xtick.major.width'] = 0.6
    matplotlib.rcParams['ytick.major.width'] = 0.6
    matplotlib.rcParams['xtick.minor.width'] = 0.45
    matplotlib.rcParams['ytick.minor.width'] = 0.45

    # Axis line width
    matplotlib.rcParams['axes.linewidth'] = 0.6

    # Line styles
    matplotlib.rcParams['lines.dash_capstyle'] = "round"
    matplotlib.rcParams['lines.solid_capstyle'] = "round"
    matplotlib.rcParams['legend.handletextpad'] = 0.4
    matplotlib.rcParams['legend.handlelength'] = 2

    # LaTeX rendering (optional - only if user has LaTeX installed)
    try:
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{cmbright}  \usepackage[T1]{fontenc}')
    except (RuntimeError, FileNotFoundError):
        # LaTeX not installed or not configured
        pass  # Use default fonts silently


def plot_wind_solution(solution, show_hot_only=True, show_clouds=True, figsize=(4, 7)):
    """
    Create the standard multi-panel plot showing wind solution.

    Parameters
    ----------
    solution : Solution object
        The solution from WindModel.run()
    show_hot_only : bool
        Whether to show the hot-only comparison
    show_clouds : bool
        Whether to show individual cloud species
    figsize : tuple
        Figure size in inches
    """
    setup_plotting_style()

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Get cloud colors
    if show_clouds:
        cloud_colors = get_cloud_colors(solution.model.N_cloud_species)

    # Panel 1: Velocity
    ax1.loglog(solution.r, solution.v, 'k-', lw=1.5, label=r'$v_{\rm wind}$')
    if show_clouds:
        for i in range(solution.model.N_cloud_species):
            # Mask cloud velocities where cloud mass is below minimum
            v_cl_masked = mask_cloud_data(solution.v_cl[i], solution.M_clouds[i], 
                                         solution.model.config.M_cloud_min)
            ax1.loglog(solution.r, v_cl_masked, color=cloud_colors[i], lw=1)
    if show_hot_only:
        ax1.loglog(solution.r_hot, solution.v_hot, '-', color='grey', lw=1,
                   label=r'$v_{\rm hot,\,only}$')

    # Sound speed
    c_s = np.sqrt(1.67 * solution.P / solution.rho) / 1e5  # km/s
    ax1.loglog(solution.r, c_s, 'k:', lw=0.8, label=r'$c_s$')

    ax1.set_ylabel(r'$v$ [km/s]')
    ax1.set_ylim(bottom=30)
    ax1.legend(frameon=False, fontsize=8, loc='best')
    ax1.tick_params(labelbottom=False)  # Hide x-axis labels on shared axis

    # Panel 2: Mass flux
    ax2.loglog(solution.r, solution.Mdot/solution.model.SFR, 'k-', lw=1.5, label=r'$\dot{M}_{\rm wind}$')
    if show_hot_only:
        ax2.loglog(solution.r_hot, solution.Mdot_hot/solution.model.SFR, '-', color='grey', lw=1,
                   label=r'$\dot{M}_{\rm hot,\,only}$')

    # Cloud mass flux
    if show_clouds:
        Mdot_cl_total = np.zeros_like(solution.r)

        # Get injection parameters
        r0 = solution.model.r_star_kpc * kpc
        injection_radius = solution.model.config.cold_cloud_injection_radial_extent_frac * r0
        injection_power = solution.model.config.cold_cloud_injection_radial_power

        for i in range(solution.model.N_cloud_species):
            # Calculate cloud number flux at each radius using injection function
            # Ndot_cloud = Ndot_cloud0 * injection_function(r)
            r_cgs = solution.sol.t  # radius in cm
            injection_function = np.where(r_cgs < injection_radius,
                                         (r_cgs/injection_radius)**injection_power,
                                         1.0)
            Ndot_cloud_i = solution.model.Ndot_cloud0[i] * injection_function

            # Cloud mass flux: Mdot_cl = Ndot_cloud * M_cloud
            # Ndot_cloud0 is in 1/s, M_clouds is in Msun
            Mdot_cl_i = Ndot_cloud_i * solution.M_clouds[i] * Msun / (Msun/yr) / solution.model.SFR

            # Mask where clouds don't exist
            Mdot_cl_i_masked = mask_cloud_data(Mdot_cl_i, solution.M_clouds[i],
                                              solution.model.config.M_cloud_min)
            ax2.loglog(solution.r, Mdot_cl_i_masked, '-', color=cloud_colors[i], lw=0.8)
            # Add to total (use filled values with 0 for masked regions)
            Mdot_cl_total += np.ma.filled(Mdot_cl_i_masked, 0)

        # Plot total cloud mass flux as dashed black line
        Mdot_cl_total_masked = np.ma.masked_where(Mdot_cl_total <= 0, Mdot_cl_total)
        ax2.loglog(solution.r, Mdot_cl_total_masked, 'k--', lw=1.2, label=r'$\dot{M}_{\rm cl,\,total}$')

    ax2.set_ylabel(r'$\dot{M}/{\rm SFR}$')

    # Set ylim to ensure room for legend
    # Find maximum of wind and cloud total mass flux
    max_mdot = np.max(solution.Mdot/solution.model.SFR)
    if show_clouds and np.any(Mdot_cl_total > 0):
        max_mdot = max(max_mdot, np.max(Mdot_cl_total_masked))

    # Set top to 3x the maximum to leave room for legend
    ax2.set_ylim(top=3*max_mdot)

    ax2.legend(frameon=False, fontsize=8, loc='upper left', ncol=3)
    ax2.tick_params(labelbottom=False)  # Hide x-axis labels on shared axis

    # Panel 3: Cloud masses
    if show_clouds:
        for i in range(solution.model.N_cloud_species):
            # M_clouds is in Msun, display directly
            M_cl_i = mask_cloud_data(solution.M_clouds[i], solution.M_clouds[i],
                                    solution.model.config.M_cloud_min)
            ax3.loglog(solution.r, M_cl_i, '-', color=cloud_colors[i], lw=0.8)

        # Add cloud mass colorbar
        M_cloud0_log = np.log10(solution.model.M_cloud0 / Msun)
        add_cloud_mass_colorbar(ax3, solution.model.N_cloud_species, M_cloud0_log)

    ax3.set_xlabel(r'$r$ [kpc]')
    ax3.set_ylabel(r'$M_{\rm cl}$ [$M_\odot$]')
    ax3.set_xlim(0.29, 30.5)

    return fig, (ax1, ax2, ax3)


def plot_profiles(solution, quantities=['velocity', 'density', 'temperature'],
                  figsize=(6, 8)):
    """
    Plot various wind profiles.

    Parameters
    ----------
    solution : Solution object
        The solution from WindModel.run()
    quantities : list
        Which quantities to plot
    figsize : tuple
        Figure size in inches
    """
    setup_plotting_style()

    n_panels = len(quantities)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                            constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for i, quantity in enumerate(quantities):
        ax = axes[i]

        if quantity == 'velocity':
            ax.loglog(solution.r, solution.v, 'k-', lw=1.5, label='Wind')
            # Plot mean cloud velocity (averaged over species)
            if hasattr(solution, 'v_cl') and solution.v_cl.ndim > 1:
                v_cl_mean = np.mean(solution.v_cl, axis=0)
                ax.loglog(solution.r, v_cl_mean, 'k--', lw=1, label='Clouds (mean)')
            ax.set_ylabel(r'$v$ [km/s]')

        elif quantity == 'density':
            ax.loglog(solution.r, solution.n, 'k-', lw=1.5, label='Hot phase')
            ax.set_ylabel(r'$n$ [cm$^{-3}$]')

        elif quantity == 'temperature':
            ax.loglog(solution.r, solution.T, 'k-', lw=1.5, label='Hot phase')
            ax.axhline(solution.model.config.T_cl, color='k', ls='--', lw=1, label='Clouds')
            ax.set_ylabel(r'$T$ [K]')

        elif quantity == 'pressure':
            ax.loglog(solution.r, solution.P, 'k-', lw=1.5)
            ax.set_ylabel(r'$P$ [dyne/cm$^2$]')

        elif quantity == 'mass_flux':
            ax.loglog(solution.r, solution.Mdot/solution.model.SFR, 'k-', lw=1.5)
            ax.set_ylabel(r'$\dot{M}/{\rm SFR}$')

        elif quantity == 'metallicity':
            # Plot hot wind metallicity
            ax.semilogx(solution.r, solution.Z, 'k-', lw=1.5, label='Hot wind')
            # Plot mean cloud metallicity if available
            if hasattr(solution, 'Z_cl') and solution.Z_cl.ndim > 1:
                Z_cl_mean = np.mean(solution.Z_cl, axis=0)
                ax.semilogx(solution.r, Z_cl_mean, 'k--', lw=1, label='Clouds (mean)')
            ax.set_ylabel(r'$Z/Z_\odot$')

        ax.legend(frameon=False, fontsize=8, loc='best')

        if i < n_panels - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel(r'$r$ [kpc]')
    axes[-1].set_xlim(solution.r[0], solution.r[-1])

    return fig, axes



def plot_column_density_distribution(solution, cloud_index=None,
                                   figsize=(5, 4),
                                   xlim=None, ylim=None,
                                   log_scale=False,
                                   show_species=False,
                                   species_alpha=0.5,
                                   show_moments=False,
                                   **kwargs):
    """
    Plot column density distribution dN/dv in units of cm^-2 / (km/s).

    This is the standard format for absorption line observations.

    Parameters
    ----------
    solution : Solution object
        The wind solution from WindModel.run()
    cloud_index : int, optional
        Index of specific cloud species. If None, sum over all.
    figsize : tuple
        Figure size in inches
    xlim : tuple, optional
        Velocity axis limits [km/s]
    ylim : tuple, optional
        Column density axis limits
    log_scale : bool
        Whether to use log scale for y-axis
    show_species : bool
        Whether to show individual cloud species contributions
    species_alpha : float
        Transparency for individual species lines
    show_moments : bool
        Whether to show mean and dispersion on plot
    **kwargs : dict
        Additional arguments for calculate_column_density_distribution

    Returns
    -------
    fig, ax : matplotlib objects
    """
    setup_plotting_style()

    if show_species and cloud_index is None:
        # Calculate for all species
        from .observables import calculate_column_density_by_species
        v_cloud, dN_dv_dict = calculate_column_density_by_species(solution, **kwargs)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        # Get cloud colors
        cloud_colors = get_cloud_colors(solution.model.N_cloud_species)

        # Plot individual species
        for i, dN_dv_i in enumerate(dN_dv_dict['species']):
            dN_dv_i_plot = np.ma.masked_less_equal(dN_dv_i, 0.0)
            ax.plot(v_cloud, dN_dv_i_plot, '-', color=cloud_colors[i],
                    alpha=species_alpha, lw=1)

        # Plot total
        dN_dv_total_plot = np.ma.masked_less_equal(dN_dv_dict['total'], 0.0)
        ax.plot(v_cloud, dN_dv_total_plot, 'k-', lw=1.5, label='Total')

        # Add cloud mass colorbar similar to plot_wind_solution
        M_cloud0_log = np.log10(dN_dv_dict['M_cloud0'])
        add_cloud_mass_colorbar(ax, solution.model.N_cloud_species, M_cloud0_log,
                               width="40%", height="4%", fontsize=5, title_fontsize=6)
        
        # Add small legend just for "Total" line
        ax.legend(frameon=False, fontsize=8, loc='upper right')

    else:
        # Original single distribution
        from .observables import calculate_column_density_distribution
        v_cloud, dN_dv_column = calculate_column_density_distribution(solution,
                                                                     cloud_index=cloud_index,
                                                                     **kwargs)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        # Plot distribution
        dN_dv_plot = np.ma.masked_less_equal(dN_dv_column, 0.0)
        ax.plot(v_cloud, dN_dv_plot, 'k-', lw=1.5)

    # Calculate and show moments if requested
    if show_moments:
        # Get the appropriate dN/dv data
        if show_species and cloud_index is None:
            dN_dv_for_moments = dN_dv_dict['total']
        else:
            dN_dv_for_moments = dN_dv_column

        # Calculate moments
        from .observables import calculate_velocity_moments
        moments = calculate_velocity_moments(v_cloud, dN_dv_for_moments)

        if moments['raw'][0] > 0:
            mean_v = moments.get('mean', 0)
            disp_v = moments.get('dispersion', 0)

            # Add vertical lines for mean ± dispersion
            ax.axvline(mean_v, color='k', ls='--', lw=1, alpha=0.5)
            if disp_v > 0:
                ax.axvline(mean_v - disp_v, color='k', ls=':', lw=1, alpha=0.5)
                ax.axvline(mean_v + disp_v, color='k', ls=':', lw=1, alpha=0.5)

            # Add text annotation
            ax.text(0.95, 0.95,
                    f'$\\langle v \\rangle = {mean_v:.0f}$ km/s\n' +
                    f'$\\sigma_v = {disp_v:.0f}$ km/s',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set scales
    if log_scale:
        ax.set_yscale('log')

    # Labels with proper units
    ax.set_xlabel(r'$v$ [km s$^{-1}$]')
    ax.set_ylabel(r'$dN/dv$ [cm$^{-2}$ (km s$^{-1}$)$^{-1}$]')

    # Set limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, None)

    if ylim is not None:
        ax.set_ylim(ylim)
    elif not log_scale:
        ax.set_ylim(0, None)

    return fig, ax
