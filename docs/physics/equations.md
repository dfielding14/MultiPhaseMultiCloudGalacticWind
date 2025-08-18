# Governing Equations

## Hot Wind Evolution

The hot wind phase evolves according to conservation of mass, momentum, and energy:

### Mass Conservation

$$\frac{d}{dr}(4\pi r^2 \rho_h v_h) = 4\pi r^2 \left( \dot{\rho}_{inj} - \dot{\rho}_{mix} \right)$$

where:
- $\dot{\rho}_{inj}$ = mass injection rate density
- $\dot{\rho}_{mix}$ = mass exchange with clouds

### Momentum Conservation

$$\rho_h v_h \frac{dv_h}{dr} = -\frac{dP}{dr} - \rho_h \frac{v_{circ}^2}{r} + \dot{\rho}_{inj} v_{inj} - \dot{\rho}_{mix} v_h + F_{drag}$$

where:
- $P$ = thermal pressure
- $v_{circ}$ = circular velocity (gravity)
- $F_{drag}$ = momentum transfer to clouds

### Energy Conservation

$$\frac{d}{dr}\left[ 4\pi r^2 \rho_h v_h \left( \frac{v_h^2}{2} + \frac{\gamma}{\gamma-1} \frac{P}{\rho_h} \right) \right] = 4\pi r^2 \left( \dot{E}_{inj} - \dot{E}_{cool} - \dot{E}_{mix} \right)$$

where:
- $\dot{E}_{inj}$ = energy injection rate
- $\dot{E}_{cool}$ = radiative cooling rate
- $\dot{E}_{mix}$ = energy loss to mixing

## Cloud Evolution

Each cloud species $i$ evolves independently:

### Cloud Mass Evolution

$$\frac{dM_{c,i}}{dr} = \frac{1}{v_{c,i}} \left( \dot{M}_{mix,i} + \dot{M}_{inj,i} \right)$$

where:
- $\dot{M}_{mix,i} = 4\pi R_{c,i}^2 \rho_{mix} v_{turb,i}$ (mass growth from mixing)
- $\dot{M}_{inj,i}$ = injection rate for species $i$

### Cloud Velocity Evolution

$$v_{c,i} \frac{dv_{c,i}}{dr} = -\frac{v_{circ}^2}{r} + a_{drag,i}$$

where the drag acceleration is:
$$a_{drag,i} = \frac{3 C_d \rho_h (v_h - v_{c,i}) |v_h - v_{c,i}|}{4 \rho_c R_{c,i}}$$

### Cloud Metallicity Evolution

$$\frac{dZ_{c,i}}{dr} = \frac{1}{v_{c,i} M_{c,i}} \left[ \dot{M}_{mix,i} (Z_h - Z_{c,i}) + \dot{M}_{inj,i} (Z_{inj} - Z_{c,i}) \right]$$

## Normalized Form

For numerical stability, we solve the equations in normalized form:

### Hot Wind ODEs

$$\frac{dv_h}{dr} = \frac{v_h/r}{1 - 1/\mathcal{M}^2} \left[ \frac{2}{\mathcal{M}^2} - \left(\frac{v_{circ}}{v_h}\right)^2 - \mathcal{S}_v \right]$$

$$\frac{d\rho_h}{dr} = -\frac{\rho_h}{v_h} \frac{dv_h}{dr} - \frac{2\rho_h}{r} + \frac{\mathcal{S}_\rho}{v_h}$$

$$\frac{dP}{dr} = -\frac{\gamma P}{v_h} \frac{dv_h}{dr} - \frac{2\gamma P}{r} + \frac{(\gamma-1)}{v_h} \mathcal{S}_E + \frac{\gamma P}{\rho_h v_h} \mathcal{S}_\rho$$

$$\frac{d(\rho_h Z_h)}{dr} = Z_h \frac{d\rho_h}{dr} + \frac{\mathcal{S}_Z}{v_h}$$

where the source terms are:

$$\mathcal{S}_v = \frac{\dot{\rho}_{inj} v_{inj} - \sum_i \dot{\rho}_{mix,i} v_h + \sum_i F_{drag,i}}{\rho_h v_h^2}$$

$$\mathcal{S}_\rho = \dot{\rho}_{inj} - \sum_i \dot{\rho}_{mix,i}$$

$$\mathcal{S}_E = \dot{E}_{inj} - \dot{E}_{cool} - \sum_i \dot{E}_{mix,i}$$

$$\mathcal{S}_Z = \dot{\rho}_{inj} Z_{inj} - \sum_i \dot{\rho}_{mix,i} Z_h$$

## Power-Law Cloud Distribution

The cloud mass distribution follows:

$$\frac{dN}{dM} = N_0 M^{-\alpha}$$

For a finite number of species with logarithmic spacing:

$$M_i = M_{min} \left( \frac{M_{max}}{M_{min}} \right)^{(i-1)/(N-1)}$$

The number of clouds per species:

$$N_i = \int_{M_{i,low}}^{M_{i,high}} \frac{dN}{dM} dM$$

## TRML Physics

### Turbulent Velocity

$$v_{turb} = f_{turb} \cdot |v_h - v_c|$$

where $f_{turb} \sim 0.1 - 0.2$ is the turbulent velocity fraction.

### Mixing Layer Density

In pressure equilibrium:

$$\rho_{mix} = \sqrt{\rho_h \rho_c} = \rho_h \sqrt{\frac{T_h}{T_c}}$$

### Mass Transfer Rate

$$\dot{M}_{mix} = 4\pi R_c^2 \rho_{mix} v_{turb}$$

### Energy Dissipation

Kinetic energy dissipation:

$$\dot{E}_{diss} = \frac{1}{2} \dot{M}_{mix} (v_h - v_c)^2$$

This energy is radiated away in the mixing layer.

## Cooling Function

The cooling rate per unit volume:

$$\mathcal{L} = n_H^2 \Lambda(T, Z)$$

where $\Lambda(T, Z)$ is the cooling function from Wiersma+09 tables.

Cooling time:

$$t_{cool} = \frac{3 k_B T}{2 n_H \Lambda(T, Z)}$$

## Initial Conditions

At the injection radius $r_0$:

### Hot Wind
- Velocity: $v_{h,0} = 0.1 \times c_s$ (subsonic)
- Density: From mass injection rate
- Pressure: From energy injection rate
- Metallicity: $Z_{h,0} = Z_{inj}$

### Clouds
- Velocity: $v_{c,0} = v_{h,0}$ (comoving initially)
- Mass: Power-law distribution
- Temperature: $T_c = 10^4$ K
- Metallicity: $Z_{c,0} = Z_{inj}$

## Boundary Conditions

### Inner Boundary (r = r₀)
- Mass injection: $\dot{M} = \eta_M \times \text{SFR}$
- Energy injection: $\dot{E} = \eta_E \times 10^{51} \text{ erg} \times \text{SFR} / (100 \text{ yr})$
- Momentum injection: $\dot{p} = \dot{M} \times v_{inj}$

### Outer Boundary
- Free outflow (no boundary conditions)
- Integration terminates at events or r_max

## Non-Dimensional Parameters

Key dimensionless parameters:

- **Mach number**: $\mathcal{M} = v_h / c_s$
- **Cooling parameter**: $t_{cool} / t_{flow}$
- **Drag parameter**: $C_d \rho_h R_c / \rho_c$
- **Loading parameters**: $\eta_M$, $\eta_{M,cold}$, $\eta_E$