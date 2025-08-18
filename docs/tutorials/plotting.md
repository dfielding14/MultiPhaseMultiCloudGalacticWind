# Tutorial: Visualization and Plotting

## Introduction

This tutorial covers creating publication-quality visualizations of wind model results.

## Basic Plotting Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel
from multiphasegalacticwind.plotting import (
    plot_wind_solution, 
    plot_cloud_evolution,
    plot_column_density_distribution
)

# Configure matplotlib for publication (clean style)
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'figure.dpi': 100,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

# Use LaTeX if available
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
except:
    print("LaTeX not available, using default fonts")
```

## Using Built-in Plotting Functions

### Standard Wind Solution Plot

```python
# Create and run model
model = WindModel(SFR=10.0, v_circ=200.0, eta_M=0.1, eta_E=1.0)
solution = model.run()

# Use built-in plotting
fig, axes = plot_wind_solution(solution)
fig.suptitle('Standard Wind Solution', fontsize=16)
plt.tight_layout()
plt.show()

# Save high-quality figure
fig.savefig('wind_solution.pdf', dpi=300, bbox_inches='tight')
```

### Cloud Evolution Plot

```python
# Model with clouds
from multiphasegalacticwind import WindConfig

config = WindConfig(N_cloud_species=10)
model = WindModel(config=config, SFR=10.0, eta_M_cold=2.0)
solution = model.run()

# Plot cloud evolution
fig, axes = plot_cloud_evolution(solution)
plt.show()
```

## Custom Visualizations

### Phase Space Diagram

```python
fig, ax = plt.subplots(figsize=(10, 8))

# Hot wind
ax.scatter(solution.v, solution.T, c=solution.r, 
          s=50, cmap='hot', alpha=0.7, label='Hot wind')

# Add colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Radius [kpc]', fontsize=12)

ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('Temperature [K]')
ax.set_yscale('log')
ax.set_title('Phase Space Evolution')
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()
```

### Radial Profiles with Shading

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Velocity with Mach number regions
ax = axes[0, 0]
ax.plot(solution.r, solution.v, 'b-', linewidth=2)
ax.axhspan(0, 100, alpha=0.2, color='red', label='Subsonic')
ax.axhspan(100, 300, alpha=0.2, color='yellow', label='Transonic')
ax.axhspan(300, 1000, alpha=0.2, color='green', label='Supersonic')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Velocity [km/s]')
ax.set_title('Velocity Profile')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Density with power-law fit
ax = axes[0, 1]
ax.loglog(solution.r, solution.rho, 'r-', linewidth=2, label='Model')
# Fit power law to outer region
r_fit = solution.r[solution.r > 10]
rho_fit = solution.rho[solution.r > 10]
coeffs = np.polyfit(np.log10(r_fit), np.log10(rho_fit), 1)
rho_powerlaw = 10**(coeffs[0] * np.log10(solution.r) + coeffs[1])
ax.loglog(solution.r, rho_powerlaw, 'k--', alpha=0.5, 
         label=f'$\\rho \\propto r^{{{coeffs[0]:.1f}}}$')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Density [g/cm³]')
ax.set_title('Density Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Temperature with cooling time
ax = axes[1, 0]
ax2 = ax.twinx()
ax.loglog(solution.r, solution.T, 'g-', linewidth=2, label='Temperature')
# Calculate cooling time
from multiphasegalacticwind.cooling import tcool_P
t_cool = [tcool_P(T, P/1.38e-16, 1.0, 0.0, 0.62) 
          for T, P in zip(solution.T, solution.P)]
ax2.loglog(solution.r, np.array(t_cool)/3.15e13, 'b--', 
          linewidth=2, alpha=0.7, label='t_cool')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Temperature [K]', color='g')
ax2.set_ylabel('Cooling Time [Myr]', color='b')
ax.set_title('Temperature and Cooling')
ax.grid(True, alpha=0.3)

# Mass flux
ax = axes[1, 1]
ax.loglog(solution.r, solution.mass_flux, 'm-', linewidth=2)
ax.axhline(model.SFR, color='k', linestyle='--', alpha=0.5, 
          label=f'SFR = {model.SFR} Msun/yr')
ax.fill_between(solution.r, solution.mass_flux, model.SFR,
                where=(solution.mass_flux > model.SFR),
                alpha=0.3, color='green', label='Mass-loaded')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Mass Flux [Msun/yr]')
ax.set_title('Mass Flux')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Column Density Visualization

### Multi-Panel Column Density

```python
# Calculate observables
v_cloud, dN_dv = solution.calculate_column_density_distribution()
v_cloud_species, dN_dv_species = solution.calculate_column_density_by_species()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Total column density
ax = axes[0, 0]
ax.semilogy(v_cloud, dN_dv, 'b-', linewidth=2)
ax.fill_between(v_cloud, dN_dv, alpha=0.3)
ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('dN/dv [cm⁻² / (km/s)]')
ax.set_title('Total Column Density')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)

# Individual species
ax = axes[0, 1]
for i in range(min(5, config.N_cloud_species)):
    ax.semilogy(v_cloud_species, dN_dv_species['species'][i], 
               alpha=0.7, label=f'Species {i+1}')
ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('dN/dv [cm⁻² / (km/s)]')
ax.set_title('By Cloud Species')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)

# Cumulative distribution
ax = axes[1, 0]
cumulative = np.cumsum(dN_dv) * np.gradient(v_cloud)
ax.plot(v_cloud, cumulative / cumulative[-1], 'g-', linewidth=2)
ax.axhline(0.5, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('Cumulative Distribution')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)

# Velocity moments
ax = axes[1, 1]
moments = solution.calculate_velocity_moments()
ax.text(0.1, 0.8, f"Mean: {moments['mean']:.1f} km/s", 
        transform=ax.transAxes, fontsize=14)
ax.text(0.1, 0.6, f"Dispersion: {moments['dispersion']:.1f} km/s", 
        transform=ax.transAxes, fontsize=14)
ax.text(0.1, 0.4, f"Skewness: {moments['skewness']:.2f}", 
        transform=ax.transAxes, fontsize=14)
ax.text(0.1, 0.2, f"Kurtosis: {moments['kurtosis']:.2f}", 
        transform=ax.transAxes, fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Velocity Moments')

plt.tight_layout()
plt.show()
```

## Comparison Plots

### Multiple Models

```python
# Run models with different parameters
models = [
    {'label': 'Fiducial', 'eta_E': 1.0, 'color': 'b'},
    {'label': 'High Energy', 'eta_E': 3.0, 'color': 'r'},
    {'label': 'Low Energy', 'eta_E': 0.3, 'color': 'g'}
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for params in models:
    model = WindModel(SFR=10.0, v_circ=200.0, eta_E=params['eta_E'])
    solution = model.run()
    
    # Velocity
    axes[0].plot(solution.r, solution.v, 
                color=params['color'], 
                label=params['label'], 
                linewidth=2)
    
    # Column density
    v_cloud, dN_dv = solution.calculate_column_density_distribution()
    axes[1].semilogy(v_cloud, dN_dv, 
                    color=params['color'], 
                    label=params['label'], 
                    linewidth=2)

axes[0].set_xlabel('Radius [kpc]')
axes[0].set_ylabel('Velocity [km/s]')
axes[0].set_title('Velocity Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Velocity [km/s]')
axes[1].set_ylabel('dN/dv [cm⁻² / (km/s)]')
axes[1].set_title('Column Density Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 600)

plt.tight_layout()
plt.show()
```

## Animation

### Animated Parameter Evolution

```python
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, ax = plt.subplots(figsize=(10, 6))

eta_E_values = np.linspace(0.1, 3.0, 20)
line, = ax.plot([], [], 'b-', linewidth=2)
ax.set_xlim(0, 50)
ax.set_ylim(0, 800)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Velocity [km/s]')
ax.grid(True, alpha=0.3)

def init():
    line.set_data([], [])
    return line,

def animate(frame):
    eta_E = eta_E_values[frame]
    model = WindModel(SFR=10.0, v_circ=200.0, eta_E=eta_E, rtol=1e-6)
    solution = model.run()
    
    line.set_data(solution.r, solution.v)
    ax.set_title(f'Wind Solution: $\\eta_E$ = {eta_E:.2f}')
    return line,

anim = FuncAnimation(fig, animate, init_func=init, 
                    frames=len(eta_E_values), 
                    interval=200, blit=True)

# Save as GIF
anim.save('wind_evolution.gif', writer='pillow', fps=5)
```

## Publication Figures

### Multi-Panel Figure for Paper

```python
# Create publication-quality figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Run model
model = WindModel(SFR=10.0, v_circ=200.0, eta_M_cold=2.0)
solution = model.run()

# Panel 1: Velocity
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(solution.r, solution.v, 'b-', linewidth=2)
ax1.set_xlabel('r [kpc]')
ax1.set_ylabel('v [km s$^{-1}$]')
ax1.set_title('(a) Velocity', loc='left')
ax1.grid(True, alpha=0.3)

# Panel 2: Density
ax2 = fig.add_subplot(gs[0, 1])
ax2.loglog(solution.r, solution.rho, 'r-', linewidth=2)
ax2.set_xlabel('r [kpc]')
ax2.set_ylabel('$\\rho$ [g cm$^{-3}$]')
ax2.set_title('(b) Density', loc='left')
ax2.grid(True, alpha=0.3)

# Panel 3: Temperature
ax3 = fig.add_subplot(gs[0, 2])
ax3.loglog(solution.r, solution.T, 'g-', linewidth=2)
ax3.set_xlabel('r [kpc]')
ax3.set_ylabel('T [K]')
ax3.set_title('(c) Temperature', loc='left')
ax3.grid(True, alpha=0.3)

# Panel 4: Mach number
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(solution.r, solution.Mach, 'm-', linewidth=2)
ax4.axhline(1, color='k', linestyle='--', alpha=0.5)
ax4.set_xlabel('r [kpc]')
ax4.set_ylabel('$\\mathcal{M}$')
ax4.set_title('(d) Mach Number', loc='left')
ax4.grid(True, alpha=0.3)

# Panel 5: Mass flux
ax5 = fig.add_subplot(gs[1, 1])
ax5.loglog(solution.r, solution.mass_flux, 'c-', linewidth=2)
ax5.set_xlabel('r [kpc]')
ax5.set_ylabel('$\\dot{M}$ [M$_\\odot$ yr$^{-1}$]')
ax5.set_title('(e) Mass Flux', loc='left')
ax5.grid(True, alpha=0.3)

# Panel 6: Column density
ax6 = fig.add_subplot(gs[1, 2])
v_cloud, dN_dv = solution.calculate_column_density_distribution()
ax6.semilogy(v_cloud, dN_dv, 'b-', linewidth=2)
ax6.set_xlabel('v [km s$^{-1}$]')
ax6.set_ylabel('dN/dv [cm$^{-2}$ (km/s)$^{-1}$]')
ax6.set_title('(f) Column Density', loc='left')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 500)

# Panel 7-9: Parameter comparison
ax7 = fig.add_subplot(gs[2, :])
eta_E_values = [0.5, 1.0, 2.0]
colors = ['blue', 'green', 'red']
for eta_E, color in zip(eta_E_values, colors):
    model_comp = WindModel(SFR=10.0, v_circ=200.0, eta_E=eta_E)
    sol_comp = model_comp.run()
    ax7.plot(sol_comp.r, sol_comp.v, color=color, 
            label=f'$\\eta_E$ = {eta_E}', linewidth=2)
ax7.set_xlabel('r [kpc]')
ax7.set_ylabel('v [km s$^{-1}$]')
ax7.set_title('(g) Energy Loading Comparison', loc='left')
ax7.legend(loc='lower right')
ax7.grid(True, alpha=0.3)

plt.suptitle('Multiphase Wind Model Results', fontsize=16, y=1.02)
plt.savefig('publication_figure.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

## Color Maps and Styles

### Using Custom Colormaps

```python
import cmasher as cmr  # Better colormaps

# 2D parameter study visualization
X, Y = np.meshgrid(np.log10(eta_M_values), np.log10(eta_E_values))
Z = v_terminal_grid

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Continuous colormap
ax = axes[0]
im = ax.pcolormesh(X, Y, Z.T, cmap=cmr.cosmic, shading='auto')
ax.set_xlabel('log $\\eta_M$')
ax.set_ylabel('log $\\eta_E$')
ax.set_title('Terminal Velocity [km/s]')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('v$_{terminal}$ [km/s]')

# Discrete levels
ax = axes[1]
levels = [100, 200, 300, 400, 500, 600]
cs = ax.contourf(X, Y, Z.T, levels=levels, cmap=cmr.ember)
ax.contour(X, Y, Z.T, levels=levels, colors='white', 
          linewidths=0.5, alpha=0.5)
ax.set_xlabel('log $\\eta_M$')
ax.set_ylabel('log $\\eta_E$')
ax.set_title('Terminal Velocity Contours')
cbar = plt.colorbar(cs, ax=ax)
cbar.set_label('v$_{terminal}$ [km/s]')

plt.tight_layout()
plt.show()
```

## Best Practices

1. **Consistent style**: Use same colors/markers for same quantities
2. **Clear labels**: Always label axes with units
3. **Legends**: Include when multiple datasets
4. **Grid lines**: Add with alpha=0.3 for reference
5. **Save vectors**: Use PDF/SVG for publication
6. **Color blind friendly**: Use colorbrewer palettes
7. **Font sizes**: Scale for target medium

## See Also

- [API: Plotting](../api/plotting.md) - Plotting function reference
- [Examples](../getting_started/examples.md) - More examples
- [Parameter Studies](parameter_study.md) - Systematic exploration