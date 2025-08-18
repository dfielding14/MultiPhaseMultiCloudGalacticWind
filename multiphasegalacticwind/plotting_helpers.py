"""
Helper functions for plotting to reduce code duplication.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Try to import cmasher for colormaps
try:
    import cmasher as cmr
    HAS_CMASHER = True
except ImportError:
    HAS_CMASHER = False
    

def get_cloud_colors(n_species, cmap_name='cmr.guppy'):
    """
    Get color palette for cloud species.
    
    Parameters
    ----------
    n_species : int
        Number of cloud species
    cmap_name : str
        Name of colormap to use
        
    Returns
    -------
    colors : array
        Array of colors for each species
    """
    if HAS_CMASHER and 'cmr.' in cmap_name:
        return cmr.take_cmap_colors(cmap_name, n_species,
                                   cmap_range=(0.0, 1.0), return_fmt='hex')
    else:
        # Fall back to matplotlib colormap
        return plt.cm.viridis(np.linspace(0, 1, n_species))


def add_cloud_mass_colorbar(ax, n_species, M_cloud0_log, 
                           width="50%", height="5%", loc='lower left',
                           bbox_anchor=(0.05, 0.05, 1, 1),
                           fontsize=6, title_fontsize=7):
    """
    Add a colorbar showing cloud mass distribution.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to add colorbar to
    n_species : int
        Number of cloud species
    M_cloud0_log : array
        Log10 of initial cloud masses
    width, height : str
        Dimensions of colorbar
    loc : str
        Location of colorbar
    bbox_anchor : tuple
        Anchor position for colorbar
    fontsize : int
        Font size for labels
    title_fontsize : int
        Font size for title
        
    Returns
    -------
    cax : matplotlib axis
        The colorbar axis
    """
    # Create inset axis for colorbar
    cax = inset_axes(ax, width=width, height=height, loc=loc,
                    bbox_to_anchor=bbox_anchor, bbox_transform=ax.transAxes)
    
    # Get cloud colors
    cloud_colors = get_cloud_colors(n_species)
    
    # Fill colorbar with cloud colors
    for i in range(n_species):
        cax.axvspan(i, i+1, color=cloud_colors[i])
    
    cax.set_xlim(0, n_species)
    cax.set_ylim(0, 1)
    
    # Determine which labels to show (avoid crowding)
    label_indices = get_label_indices(n_species)
    
    # Add text labels
    for i in label_indices:
        cax.text(i + 0.5, 0.5, f'{M_cloud0_log[i]:.1f}', 
                ha='center', va='center', color='white', 
                fontsize=fontsize, fontweight='bold',
                transform=cax.transData)
    
    # Remove ticks and spines
    cax.set_xticks([])
    cax.set_yticks([])
    for spine in cax.spines.values():
        spine.set_visible(False)
    
    # Add title
    cax.text(n_species/2, 1.5, r'$\log_{10}(M_{\rm cl,0}/M_\odot)$',
            ha='center', va='bottom', transform=cax.transData, 
            fontsize=title_fontsize)
    
    return cax


def get_label_indices(n_species, max_labels=11):
    """
    Get indices for labels to show, avoiding crowding.
    
    Parameters
    ----------
    n_species : int
        Total number of species
    max_labels : int
        Maximum number of labels to show
        
    Returns
    -------
    label_indices : list
        Indices of labels to display
    """
    if n_species <= max_labels:
        return range(n_species)
    else:
        # Show approximately max_labels evenly spaced
        step = n_species / max_labels
        label_indices = [int(i * step) for i in range(max_labels)]
        # Always show last
        if n_species - 1 not in label_indices:
            label_indices[-1] = n_species - 1
        return label_indices


def mask_cloud_data(data, M_clouds, M_cloud_min):
    """
    Mask cloud data where cloud mass is below minimum.
    
    Parameters
    ----------
    data : array
        Data to mask
    M_clouds : array
        Cloud masses in Msun
    M_cloud_min : float
        Minimum cloud mass in cgs units
        
    Returns
    -------
    masked_data : masked array
        Data with mask applied
    """
    from .constants import Msun
    return np.ma.masked_where(M_clouds < M_cloud_min / Msun, data)