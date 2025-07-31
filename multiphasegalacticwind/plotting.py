"""
Plotting functions that preserve the publication-quality aesthetic from the original notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    except:
        print("LaTeX not available. Using default fonts.")


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
    if HAS_CMASHER and show_clouds:
        cloud_colors = cmr.take_cmap_colors('cmr.guppy', solution.model.N_cloud_species, 
                                           cmap_range=(0.0, 1.0), return_fmt='hex')
    else:
        cloud_colors = plt.cm.viridis(np.linspace(0, 1, solution.model.N_cloud_species))
    
    # Panel 1: Velocity
    ax1.loglog(solution.r, solution.v, 'k-', lw=1.5, label=r'$v_{\rm wind}$')
    ax1.loglog(solution.r, solution.v_cl, 'k--', lw=1, label=r'$v_{\rm clouds}$')
    if show_hot_only:
        ax1.loglog(solution.r_hot, solution.v_hot, '-', color='grey', lw=1, 
                   label=r'$v_{\rm hot,\,only}$')
    
    # Sound speed
    c_s = np.sqrt(1.67 * solution.P / solution.rho) / 1e5  # km/s
    ax1.loglog(solution.r, c_s, 'k:', lw=0.8, label=r'$c_s$')
    
    ax1.set_ylabel(r'$v$ [km/s]')
    ax1.set_ylim(bottom=30)
    ax1.legend(frameon=False, fontsize=8, loc='best')
    ax1.set_xticklabels([])
    
    # Panel 2: Mass flux
    ax2.loglog(solution.r, solution.Mdot, 'k-', lw=1.5, label=r'$\dot{M}_{\rm wind}$')
    if show_hot_only:
        ax2.loglog(solution.r_hot, solution.Mdot_hot, '-', color='grey', lw=1,
                   label=r'$\dot{M}_{\rm hot,\,only}$')
    
    # Cloud mass flux
    if show_clouds:
        for i in range(solution.model.N_cloud_species):
            Mdot_cl_i = 4 * np.pi * solution.sol.t**2 * solution.rho * solution.v_cl * 1e5 * \
                        solution.M_clouds[i] / solution.M_cloud_tot / (solution.model.SFR)
            # Mask where clouds don't exist
            Mdot_cl_i = np.ma.masked_where(solution.M_clouds[i] < 1.1*solution.model.M_cloud0[i], 
                                           Mdot_cl_i)
            ax2.loglog(solution.r, Mdot_cl_i, '-', color=cloud_colors[i], lw=0.8)
    
    ax2.set_ylabel(r'$\dot{M}/{\rm SFR}$')
    ax2.set_ylim(bottom=0.03)
    ax2.legend(frameon=False, fontsize=8, loc='best')
    ax2.set_xticklabels([])
    
    # Panel 3: Cloud masses
    if show_clouds:
        for i in range(solution.model.N_cloud_species):
            M_cl_i = np.ma.masked_where(solution.M_clouds[i] < 1.1*solution.model.M_cloud0[i],
                                       solution.M_clouds[i])
            ax3.loglog(solution.r, M_cl_i/1e3, '-', color=cloud_colors[i], lw=0.8)
        
        # Add cloud mass colorbar
        cax = inset_axes(ax3, width="50%", height="5%", loc='lower left',
                        bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax3.transAxes)
        for i in range(solution.model.N_cloud_species):
            cax.axvspan(i, i+1, color=cloud_colors[i])
        cax.set_xlim(0, solution.model.N_cloud_species)
        cax.set_xticks([0, solution.model.N_cloud_species])
        cax.set_xticklabels([r'$10^{0}$', r'$10^{5}$'], fontsize=7)
        cax.set_xlabel(r'$M_{\rm cl,0}$ [$M_\odot$]', fontsize=7)
        cax.set_yticks([])
        cax.spines['top'].set_visible(False)
        cax.spines['bottom'].set_visible(False)
        cax.spines['left'].set_visible(False)
        cax.spines['right'].set_visible(False)
    
    ax3.set_xlabel(r'$r$ [kpc]')
    ax3.set_ylabel(r'$M_{\rm cl}$ [$10^3 M_\odot$]')
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
            ax.loglog(solution.r, solution.v_cl, 'k--', lw=1, label='Clouds')
            ax.set_ylabel(r'$v$ [km/s]')
            
        elif quantity == 'density':
            ax.loglog(solution.r, solution.n, 'k-', lw=1.5, label='Hot phase')
            ax.set_ylabel(r'$n$ [cm$^{-3}$]')
            
        elif quantity == 'temperature':
            ax.loglog(solution.r, solution.T, 'k-', lw=1.5, label='Hot phase')
            ax.axhline(solution.model.T_cl, color='k', ls='--', lw=1, label='Clouds')
            ax.set_ylabel(r'$T$ [K]')
            
        elif quantity == 'pressure':
            ax.loglog(solution.r, solution.P, 'k-', lw=1.5)
            ax.set_ylabel(r'$P$ [dyne/cm$^2$]')
            
        elif quantity == 'mass_flux':
            ax.loglog(solution.r, solution.Mdot/solution.model.SFR, 'k-', lw=1.5)
            ax.set_ylabel(r'$\dot{M}/{\rm SFR}$')
            
        elif quantity == 'metallicity':
            ax.semilogx(solution.r, solution.Z_cl, 'k-', lw=1.5)
            ax.set_ylabel(r'$Z_{\rm cl}/Z_\odot$')
            
        ax.legend(frameon=False, fontsize=8, loc='best')
        
        if i < n_panels - 1:
            ax.set_xticklabels([])
    
    axes[-1].set_xlabel(r'$r$ [kpc]')
    axes[-1].set_xlim(solution.r[0], solution.r[-1])
    
    return fig, axes