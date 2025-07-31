import numpy as np
import glob
from scipy import integrate, interpolate
import matplotlib
import matplotlib.pyplot as plt

# plt.style.use('dark_background')

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = "round"
matplotlib.rcParams['lines.solid_capstyle'] = "round"
matplotlib.rcParams['legend.handletextpad'] = 0.4
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams['ytick.major.width'] = 0.6
matplotlib.rcParams['xtick.major.width'] = 0.6
matplotlib.rcParams['ytick.minor.width'] = 0.45
matplotlib.rcParams['xtick.minor.width'] = 0.45
matplotlib.rcParams['ytick.major.size'] = 2.75
matplotlib.rcParams['xtick.major.size'] = 2.75
matplotlib.rcParams['ytick.minor.size'] = 1.75
matplotlib.rcParams['xtick.minor.size'] = 1.75
matplotlib.rcParams['legend.handlelength'] = 2
matplotlib.rcParams["figure.dpi"] = 200


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{cmbright}  \usepackage[T1]{fontenc}')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
# import palettable  # Replaced with cmasher
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy import optimize
import cmasher as cmr
import matplotlib.font_manager
from matplotlib.lines import Line2D

gamma   = 5/3.
kb      = 1.3806488e-16
mp      = 1.67373522381e-24
km      = 1e5
s       = 1
yr      = 3.1536e7
Myr     = 3.1536e13
Gyr     = 3.1536e16
pc      = 3.086e18
kpc     = 1.0e3 * pc
Mpc     = 1.0e6 * pc
H0   = 67.74*km/s/Mpc
Om   = 0.3075
OL = 1 - Om
G       = 6.673e-8
Msun    = 2.e33
fb      = 0.158
keV     = 1.60218e-9

mu = 0.62
metallicity = 10**-0.5
muH = 1/0.75
redshift = 0.5


"""
Cooling curve as a function of density, temperature, metallicity, redshift
"""
file = glob.glob('/Users/dfielding/Dropbox (Simons Foundation)/Research/SZ/ProjProf_hold/ProjProf/data/Lambda_tab_redshifts.npz')
if len(file) > 0:
    data = np.load(file[0])
    Lambda_tab = data['Lambda_tab']
    redshifts  = data['redshifts']
    Zs         = data['Zs']
    log_Tbins  = data['log_Tbins']
    log_nHbins = data['log_nHbins']    
    Lambda     = interpolate.RegularGridInterpolator((log_nHbins,log_Tbins,Zs,redshifts), Lambda_tab, bounds_error=False, fill_value=1e-30)
else:
    files = np.sort(glob.glob('/Users/dfielding/Dropbox (Simons Foundation)/Research/Tables/CoolingTables/z_*hdf5'))
    redshifts = np.array([float(f[-10:-5]) for f in files])
    HHeCooling = {}
    ZCooling   = {}
    TE_T_n     = {}
    for i in range(len(files)):
        f            = h5py.File(files[i], 'r')
        i_X_He       = -3 
        Metal_free   = f.get('Metal_free')
        Total_Metals = f.get('Total_Metals')
        log_Tbins    = np.array(np.log10(Metal_free['Temperature_bins']))
        log_nHbins   = np.array(np.log10(Metal_free['Hydrogen_density_bins']))
        Cooling_Metal_free       = np.array(Metal_free['Net_Cooling'])[i_X_He] ##### what Helium_mass_fraction to use    Total_Metals = f.get('Total_Metals')
        Cooling_Total_Metals     = np.array(Total_Metals['Net_cooling'])
        HHeCooling[redshifts[i]] = interpolate.RectBivariateSpline(log_Tbins,log_nHbins, Cooling_Metal_free)
        ZCooling[redshifts[i]]   = interpolate.RectBivariateSpline(log_Tbins,log_nHbins, Cooling_Total_Metals)
        f.close()
    Lambda_tab  = np.array([[[[HHeCooling[zz].ev(lT,ln)+Z*ZCooling[zz].ev(lT,ln) for zz in redshifts] for Z in Zs] for lT in log_Tbins] for ln in log_nHbins])
    np.savez('./data/Lambda_tab_redshifts.npz', Lambda_tab=Lambda_tab, redshifts=redshifts, Zs=Zs, log_Tbins=log_Tbins, log_nHbins=log_nHbins)
    Lambda      = interpolate.RegularGridInterpolator((log_nHbins,log_Tbins,Zs,redshifts), Lambda_tab, bounds_error=False, fill_value=0)
print("interpolated lambda")


metallicity = 1.
redshift = 0.
Ps = np.logspace(-8,10,100)
rhos = np.logspace(-10,5,101)*mu*mp
Lambda_P_rho_tab = np.zeros((len(Ps),len(rhos)))
for i in range(len(Ps)):
    for j in range(len(rhos)):
        rho = rhos[j]
        T = Ps[i] * (mu*mp/rho)
        if rho > 1*muH*mp:
            rho = 1.*muH*mp
        elif rho < 1e-8*muH*mp:
            rho = 1e-8*muH*mp
        if T > 10**8.98:
            T = 10**8.98
        elif T < 1e2:
            T = 1e2
        try:
            Lambda_P_rho_tab[i,j] = Lambda((np.log10(rho/(muH*mp)),np.log10(T), metallicity, redshift))
        except:
            Lambda_P_rho_tab[i,j] = 1e-30
    if i%10 == 0:
        print(i)

Lambda_P_rho = interpolate.RegularGridInterpolator((Ps*kb, rhos), Lambda_P_rho_tab, bounds_error=False, fill_value=0.)


Lambda_z0  = interpolate.RegularGridInterpolator((log_nHbins,log_Tbins,Zs), Lambda_tab[...,0], bounds_error=False, fill_value=-1e-30)

def tcool_P(T,P, metallicity):
    T = np.where(T>10**8.98, 10**8.98, T)
    T = np.where(T<10**2, 10**2, T)
    nH_actual = P/T*(mu/muH)
    nH = np.where(nH_actual>1, 1, nH_actual)
    nH = np.where(nH<10**-8, 10**-8, nH)
    return 1.5 * (muH/mu)**2 * kb * T / ( nH_actual * Lambda_z0((np.log10(nH),np.log10(T), metallicity)))

def Lambda_z0_P(T,P, metallicity):
    nH = P/T*(mu/muH)
    if nH > 0.9:
        nH = 0.9
    return Lambda_z0((np.log10(nH),np.log10(T), metallicity))
Lambda_z0_P  = np.vectorize(Lambda_z0_P)



T = np.logspace(3.5,6.5,1000)
T_tcool_min_array = np.zeros((len(Ps),len(Zs)))
tcool_min_array = np.zeros((len(Ps),len(Zs)))
for i,P in enumerate(Ps):
    if P > 10**4.2:
        continue
    for j,Z in enumerate(Zs):
        tcools = tcool_P(T, P, Z)
        T_tcool_min_array[i,j] = T[np.where(tcools == np.min(tcools[np.where(tcools>0)]))[0][0]]
        tcool_min_array[i,j] = np.min(tcools[tcools>0])

for i in np.where(Ps > 10**4.2)[0]:
    T_tcool_min_array[i] = T_tcool_min_array[np.where(Ps > 10**4.2)[0][0]-1]
    tcool_min_array[i] = tcool_min_array[np.where(Ps > 10**4.2)[0][0]-1] * (Ps[i]/ Ps[np.where(Ps > 10**4.2)[0][0]-1])**-1

T_tcool_min_P  = interpolate.RegularGridInterpolator((Ps,Zs), T_tcool_min_array, bounds_error=False, fill_value=None )
tcool_min_P  = interpolate.RegularGridInterpolator((Ps,Zs), tcool_min_array, bounds_error=False, fill_value=None )


def Field_Length(state, f_spitzer=1):
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    Z_wind       = rhoZ_wind/rho_wind
    T_wind       = Pressure/kb/(rho_wind/(mu*mp))
    kappa        = 5.0e-7 * T_wind**2.5
    edot_cool    = 1.5*Pressure / tcool_min_P((Pressure/kb,Z_wind/Z_solar))   
    return np.sqrt(f_spitzer*kappa*T_wind / edot_cool)

def Field_Length_mix(state, f_spitzer=1):
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    Z_wind       = rhoZ_wind/rho_wind
    T_wind       = Pressure/kb/(rho_wind/(mu*mp))
    Z_cloud      = state[6]

    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    kappa = 5.0e-7 * T_wind**2.5
    edot_cool = 1.5*Pressure / t_cool_layer   
    return np.sqrt(f_spitzer*kappa*T_wind / (edot_cool))

def cloud_radius(r, state):
    v_wind       = state[0]
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    M_cloud      = state[4]
    v_cloud      = state[5]
    Z_cloud      = state[6]
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    return r_cloud

def cloud_ksi(r, state):
    v_wind       = state[0]
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    Z_wind       = rhoZ_wind/rho_wind
    M_cloud      = state[4]
    v_cloud      = state[5]
    Z_cloud      = state[6]
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kb * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    return ksi




def Cooling_and_Acceleration(r, state):
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]
    rhoZ_wind  = state[3]
    M_cloud    = state[4]
    v_cloud    = state[5]
    Z_cloud    = state[6]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)
    Z_wind       = rhoZ_wind/rho_wind
    vc           = v_circ0 * np.where(r<r0, r/r0, 1.0)
    Phir         = v_circ0**2 * np.where(r<r0, 0.5 * (r/r0)**2, np.log(r/r0)) 
    vBsq_wind    = 0.5 * v_wind**2 + (gamma / (gamma-1)) * Pressure/rho_wind + Phir

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)
    Mdot_SN = Mdot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)

    # cloud properties
    Ndot_cloud              = Ndot_cloud0 * np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
    number_density_cloud    = Ndot_cloud/(Omwind * v_cloud * r**2)
    cs_cl_sq                = gamma * kb*T_cloud/(mu*mp)
    vBsq_cl                 = 0.5 * v_cloud**2 + (1 / (gamma-1)) * cs_cl_sq + Phir

    # cloud transfer rates
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kb * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    AreaBoost    = geometric_factor * chi**CoolingAreaChiPower
    v_turb_cold  = v_turb * chi**ColdTurbulenceChiPower
    Mdot_grow    = Mdot_coefficient * 3.0 *  M_cloud * v_turb * AreaBoost / (r_cloud * chi) * np.where( ksi < 1, ksi**0.5, ksi**0.25 )
    Mdot_loss    = Mdot_coefficient * 3.0 * -M_cloud * v_turb_cold / r_cloud 
    Mdot_cloud   = np.where(M_cloud > M_cloud_min, Mdot_grow + Mdot_loss, 0)

    # density
    drhodt      = -1.0 * (number_density_cloud * Mdot_cloud)
    drhodt_plus     = -1.0 * (number_density_cloud * Mdot_loss)
    drhodt_minus    = -1.0 * (number_density_cloud * Mdot_grow) 

    # momentum
    p_dot_drag      = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2 * np.where(M_cloud>M_cloud_min, 1, 0)
    p_dot_transfer  = v_wind*Mdot_grow + v_cloud*Mdot_loss
    dpdt            = -1.0 * (number_density_cloud * (p_dot_transfer + p_dot_drag))
    dpdt_p          = -1.0 * (number_density_cloud * v_cloud*Mdot_loss)
    dpdt_m          = -1.0 * (number_density_cloud * v_wind*Mdot_grow)
    dpdt_drag       = -1.0 * (number_density_cloud * p_dot_drag)
    
    # energy
    e_dot_cool        = 0.0 if (Cooling_Factor==0) else (rho_wind/(muH*mp))**2 * Lambda_P_rho((Pressure,rho_wind))
    e_dot_transfer    = vBsq_wind*Mdot_grow + vBsq_cl*Mdot_loss
    dedt              = -1.0 * (number_density_cloud * (e_dot_transfer + p_dot_drag*v_cloud)) - e_dot_cool
    dv_cloud_dr       = (p_dot_drag + v_rel*Mdot_grow - M_cloud * vc**2/r) / (M_cloud * v_cloud) * np.where(M_cloud>M_cloud_min, 1, 0)
    return number_density_cloud * e_dot_transfer, e_dot_cool, dv_cloud_dr


def Wind_Evo(r, state):
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]
    rhoZ_wind  = state[3]
    M_cloud    = state[4]
    v_cloud    = state[5]
    Z_cloud    = state[6]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)
    Z_wind       = rhoZ_wind/rho_wind
    vc           = v_circ0 * np.where(r<r0, r/r0, 1.0)
    Phir         = v_circ0**2 * np.where(r<r0, 0.5 * (r/r0)**2, np.log(r/r0)) 
    # vc           = v_circ0 * np.where(r<r0, (r/r0)**2, 1.0)
    # Phir         = v_circ0**2 * np.where(r<r0, (1/3.) * (r/r0)**3, np.log(r)) 
    vBsq_wind    = 0.5 * v_wind**2 + (gamma / (gamma-1)) * Pressure/rho_wind + Phir

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)
    Mdot_SN = Mdot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)

    # cloud properties
    Ndot_cloud              = Ndot_cloud0 * np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
    number_density_cloud    = Ndot_cloud/(Omwind * v_cloud * r**2)
    cs_cl_sq                = gamma * kb*T_cloud/(mu*mp)
    vBsq_cl                 = 0.5 * v_cloud**2 + (1 / (gamma-1)) * cs_cl_sq + Phir

    # cloud transfer rates
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kb * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    AreaBoost    = geometric_factor * chi**CoolingAreaChiPower
    v_turb_cold  = v_turb * chi**ColdTurbulenceChiPower
    Mdot_grow    = Mdot_coefficient * 3.0 * M_cloud * v_turb * AreaBoost / (r_cloud * chi) * np.where( ksi < 1, ksi**0.5, ksi**0.25 )
    Mdot_loss    = Mdot_coefficient * 3.0 * M_cloud * v_turb_cold / r_cloud 
    Mdot_cloud   = np.where(M_cloud > M_cloud_min, Mdot_grow - Mdot_loss, 0)

    # density
    drhodt       = (number_density_cloud * Mdot_cloud)
    drhodt_plus  = (number_density_cloud * Mdot_loss)
    drhodt_minus = (number_density_cloud * Mdot_grow) 

    # momentum
    p_dot_drag   = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2 * np.where(M_cloud>M_cloud_min, 1, 0)
    dpdt_drag    = (number_density_cloud * p_dot_drag)
    
    # energy
    e_dot_cool   = 0.0 if (Cooling_Factor==0) else (rho_wind/(muH*mp))**2 * Lambda_P_rho((Pressure,rho_wind))

    # metallicity
    drhoZdt         = -1.0 * (number_density_cloud * (Z_wind*Mdot_grow + Z_cloud*Mdot_loss))

    # wind gradients
    # velocity
    dv_dr       = 2/Mach_sq_wind
    dv_dr      += - (vc/v_wind)**2
    dv_dr      +=  drhodt_minus/(rho_wind*v_wind/r) *  (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    dv_dr      += (gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    dv_dr      += -(gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    dv_dr      += -dpdt_drag/(rho_wind*v_wind**2/r)
    dv_dr      *= (v_wind/r)/(1.0-(1.0/Mach_sq_wind))
    
    # density
    drho_dr       = -2
    drho_dr      += (vc/v_wind)**2
    drho_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    drho_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    drho_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    drho_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    drho_dr      *= (rho_wind/r)/(1.0-(1.0/Mach_sq_wind))

    # pressure
    dP_dr       = -2
    dP_dr      += (vc/v_wind)**2
    dP_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.* (v_rel**2 / cs_sq_wind)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/cs_sq_wind)
    dP_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    dP_dr      *= (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind))


    drhoZ_dr   = drho_dr*(rhoZ_wind/rho_wind) + (rhoZ_wind/r) * drhodt_plus/(rho_wind*v_wind/r) * (Z_cloud/Z_wind - 1)

    # cloud gradients
    dM_cloud_dr = Mdot_cloud/v_cloud

    dv_cloud_dr = (p_dot_drag + v_rel*Mdot_grow - M_cloud * vc**2/r) / (M_cloud * v_cloud) * np.where(M_cloud>M_cloud_min, 1, 0)

    dZ_cloud_dr = (Z_wind-Z_cloud) * Mdot_grow / (M_cloud * v_cloud) * np.where(M_cloud>M_cloud_min, 1, 0)

    return np.r_[dv_dr, drho_dr, dP_dr, drhoZ_dr, dM_cloud_dr, dv_cloud_dr, dZ_cloud_dr]


def Gradient_Components(r, state):
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]
    rhoZ_wind  = state[3]
    M_cloud    = state[4]
    v_cloud    = state[5]
    Z_cloud    = state[6]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)
    Z_wind       = rhoZ_wind/rho_wind
    vc           = v_circ0 * np.where(r<r0, r/r0, 1.0)
    Phir         = v_circ0**2 * np.where(r<r0, 0.5 * (r/r0)**2, np.log(r/r0)) 
    # vc           = v_circ0 * np.where(r<r0, (r/r0)**2, 1.0)
    # Phir         = v_circ0**2 * np.where(r<r0, (1/3.) * (r/r0)**3, np.log(r)) 
    vBsq_wind    = 0.5 * v_wind**2 + (gamma / (gamma-1)) * Pressure/rho_wind + Phir

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)
    Mdot_SN = Mdot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)

    # cloud properties
    Ndot_cloud              = Ndot_cloud0 * np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
    number_density_cloud    = Ndot_cloud/(Omwind * v_cloud * r**2)
    cs_cl_sq                = gamma * kb*T_cloud/(mu*mp)
    vBsq_cl                 = 0.5 * v_cloud**2 + (1 / (gamma-1)) * cs_cl_sq + Phir

    # cloud transfer rates
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kb * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    AreaBoost    = geometric_factor * chi**CoolingAreaChiPower
    v_turb_cold  = v_turb * chi**ColdTurbulenceChiPower
    Mdot_grow    = Mdot_coefficient * 3.0 * M_cloud * v_turb * AreaBoost / (r_cloud * chi) * np.where( ksi < 1, ksi**0.5, ksi**0.25 )
    Mdot_loss    = Mdot_coefficient * 3.0 * M_cloud * v_turb_cold / r_cloud 
    Mdot_cloud   = np.where(M_cloud > M_cloud_min, Mdot_grow - Mdot_loss, 0)

    # density
    drhodt       = (number_density_cloud * Mdot_cloud)
    drhodt_plus  = (number_density_cloud * Mdot_loss)
    drhodt_minus = (number_density_cloud * Mdot_grow) 

    # momentum
    p_dot_drag   = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2 * np.where(M_cloud>M_cloud_min, 1, 0)
    dpdt_drag    = (number_density_cloud * p_dot_drag)
    
    # energy
    e_dot_cool   = 0.0 if (Cooling_Factor==0) else (rho_wind/(muH*mp))**2 * Lambda_P_rho((Pressure,rho_wind))

    # metallicity
    drhoZdt         = -1.0 * (number_density_cloud * (Z_wind*Mdot_grow + Z_cloud*Mdot_loss))

    # wind gradients
    # velocity
    dv_dr       = 2/Mach_sq_wind
    dv_dr      += - (vc/v_wind)**2
    dv_dr      +=  drhodt_minus/(rho_wind*v_wind/r) *  (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    dv_dr      += (gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    dv_dr      += -(gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    dv_dr      += -dpdt_drag/(rho_wind*v_wind**2/r)
    dv_dr      *= (v_wind/r)/(1.0-(1.0/Mach_sq_wind))
    
    dv_dr_1      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * 2/Mach_sq_wind
    dv_dr_2      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * - (vc/v_wind)**2
    dv_dr_3      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) *  drhodt_minus/(rho_wind*v_wind/r) *  (1/Mach_sq_wind)
    dv_dr_4      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_plus/(rho_wind*v_wind/r) * (1/Mach_sq_wind)
    dv_dr_5      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dv_dr_6      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    dv_dr_7      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    dv_dr_8      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * (gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    dv_dr_9      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -(gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    dv_dr_10     = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -dpdt_drag/(rho_wind*v_wind**2/r)

    # density
    drho_dr       = -2
    drho_dr      += (vc/v_wind)**2
    drho_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    drho_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    drho_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    drho_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    drho_dr      *= (rho_wind/r)/(1.0-(1.0/Mach_sq_wind))

    drho_dr_1      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -2
    drho_dr_2      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * (vc/v_wind)**2
    drho_dr_3      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_minus/(rho_wind*v_wind/r)
    drho_dr_4      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r)
    drho_dr_5      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    drho_dr_6      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    drho_dr_7      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    drho_dr_8      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -(gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    drho_dr_9      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    drho_dr_10     = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * dpdt_drag/(rho_wind*v_wind**2/r)
    
    # pressure
    dP_dr       = -2
    dP_dr      += (vc/v_wind)**2
    dP_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.* (v_rel**2 / cs_sq_wind)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/cs_sq_wind)
    dP_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    dP_dr      *= (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind))

    dP_dr_1       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * -2
    dP_dr_2       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * (vc/v_wind)**2
    dP_dr_3       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * -drhodt_minus/(rho_wind*v_wind/r)
    dP_dr_4       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r)
    dP_dr_5       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dP_dr_6       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.* (v_rel**2 / cs_sq_wind)
    dP_dr_7       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/cs_sq_wind)
    dP_dr_8       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * -(gamma-1)*e_dot_cool/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr_9       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr_10      = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * dpdt_drag/(rho_wind*v_wind**2/r)

    # entropy
    K = (Pressure/kb) / (rho_wind/(mu*mp))**gamma
    dK_dr   = (K/r) * (gamma * drhodt_plus/(rho_wind*v_wind/r) * ((gamma-1)/2. * (v_rel**2/cs_sq_wind) - (cs_sq_wind-cs_cl_sq)/cs_sq_wind) - (r/v_wind) * (e_dot_cool)/Pressure * (gamma-1) + (r/v_wind) * dpdt_drag*v_rel/Pressure * (gamma-1))  
    dK_dr_1 = (K/r) * (gamma * drhodt_plus/(rho_wind*v_wind/r) * ((gamma-1)/2. * (v_rel**2/cs_sq_wind)))
    dK_dr_2 = (K/r) * (gamma * drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind-cs_cl_sq)/cs_sq_wind))
    dK_dr_3 = (K/r) * (-(r/v_wind) * (e_dot_cool)/Pressure * (gamma-1))
    dK_dr_4 = (K/r) * ((r/v_wind) * dpdt_drag*v_rel/Pressure * (gamma-1))

    # cloud gradients
    dM_cloud_dr   = Mdot_cloud/v_cloud* np.where(M_cloud>M_cloud_min, 1, 0)
    dM_cloud_dr_1 = Mdot_grow/v_cloud * np.where(M_cloud>M_cloud_min, 1, 0)
    dM_cloud_dr_2 = -Mdot_loss/v_cloud * np.where(M_cloud>M_cloud_min, 1, 0)

    dv_cloud_dr   = (p_dot_drag + v_rel*Mdot_grow - M_cloud * vc**2/r) / (M_cloud * v_cloud)* np.where(M_cloud>M_cloud_min, 1, 0)
    dv_cloud_dr_1 = (v_rel*Mdot_grow) / (M_cloud * v_cloud)* np.where(M_cloud>M_cloud_min, 1, 0)
    dv_cloud_dr_2 = (p_dot_drag) / (M_cloud * v_cloud)* np.where(M_cloud>M_cloud_min, 1, 0)
    dv_cloud_dr_3 = (-M_cloud * vc**2/r) / (M_cloud * v_cloud)* np.where(M_cloud>M_cloud_min, 1, 0)

    return [[dv_dr,dv_dr_1,dv_dr_2,dv_dr_3,dv_dr_4,dv_dr_5,dv_dr_6,dv_dr_7,dv_dr_8,dv_dr_9,dv_dr_10],
            [drho_dr,drho_dr_1,drho_dr_2,drho_dr_3,drho_dr_4,drho_dr_5,drho_dr_6,drho_dr_7,drho_dr_8,drho_dr_9,drho_dr_10],
            [dP_dr,dP_dr_1,dP_dr_2,dP_dr_3,dP_dr_4,dP_dr_5,dP_dr_6,dP_dr_7,dP_dr_8,dP_dr_9,dP_dr_10],
            [dK_dr,dK_dr_1,dK_dr_2,dK_dr_3,dK_dr_4],
            [dM_cloud_dr,dM_cloud_dr_1,dM_cloud_dr_2],
            [dv_cloud_dr,dv_cloud_dr_1,dv_cloud_dr_2,dv_cloud_dr_3]]


def Hot_Wind_Evo(r, state):
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)
    vc           = v_circ0 * np.where(r<r0, r/r0, 1.0)
    Phir         = v_circ0**2 * np.where(r<r0, 0.5 * (r/r0)**2, np.log(r/r0)) 
    vBsq_wind    = 0.5 * v_wind**2 + (gamma / (gamma-1)) * Pressure/rho_wind + Phir

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(r<r0, 1.0, 0.0)
    Mdot_SN = Mdot_per_Vol * np.where(r<r0, 1.0, 0.0)

    # density
    drhodt          = Mdot_SN
    
    # momentum
    dpdt            = 0 

    # energy
    dedt            = Edot_SN

    dv_dr    = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * ( 2.0/Mach_sq_wind - 1/(rho_wind*v_wind/r) * (drhodt*(gamma+1)/2. + (gamma-1)*dedt/v_wind**2)) 
    drho_dr  = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * ( -2.0 + 1/(rho_wind*v_wind/r) * (drhodt*(gamma+3)/2. + (gamma-1)*dedt/v_wind**2 - drhodt/Mach_sq_wind)) 
    dP_dr    = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * ( -2.0 + 1/(rho_wind*v_wind/r) * (drhodt + drhodt * (gamma-1)/2.*Mach_sq_wind + (gamma-1)*Mach_sq_wind*dedt/v_wind**2))

    return np.r_[dv_dr, drho_dr, dP_dr]


def supersonic(r,z):
    return z[0]/np.sqrt(gamma*z[2]/z[1]) - (1.0 + epsilon)

supersonic.terminal = True

def subsonic(r,z):
    return z[0]/np.sqrt(gamma*z[2]/z[1]) - (1.0 - epsilon)

subsonic.terminal = True

def cold_wind(r,z):
    return np.sqrt(gamma*z[2]/z[1])/np.sqrt(gamma*kb*T_cloud/(mu*mp)) - (1.0 + epsilon)

cold_wind.terminal = True


def cloud_stop(r,z):
    return z[5] - 10e5

cloud_stop.terminal = True





SFR = 20 * Msun/yr
eta_M              = 0.1
eta_M_cold_tot     = 0.2
log_M_cloud0       = 3

### METALLICITY
Z_solar = 0.02

# feedback and SF props
E_SN  = 1e51
mstar = 100*Msun
M_cloud_min = 1e-2*Msun

## model choices
CoolingAreaChiPower         =  0.5 
ColdTurbulenceChiPower      = -0.5 
TurbulentVelocityChiPower   =  0.0 
geometric_factor            = 1.0
Mdot_coefficient            = 1.0/3.0
Cooling_Factor              = 1.0
drag_coeff                  = 0.5
f_turb0                     = 10**-1.0
v_circ0                     = 150e5 # gravitational potential assuming isothermal potential

r0                 = 300*pc
Z_wind_initial     = 2.0 * Z_solar
half_opening_angle = np.pi/2
Omwind             = 4*np.pi*(1.0 - np.cos(half_opening_angle))
eta_E              = 1

Mdot        = eta_M * SFR
Edot        = eta_E * (E_SN/mstar) * SFR

# properties at r0 if no clouds + gravity
epsilon     = 1e-5
Mach0       = 1.0 + epsilon
v0          = np.sqrt(Edot/Mdot)*(1/((gamma-1)*Mach0) + 1/2.)**(-1/2.)
rho0        = Mdot/(Omwind*r0**2 * v0)
P0          = rho0*v0**2 / Mach0**2 / gamma
rhoZ0       = rho0 * Z_wind_initial
print( "v_wind = %.1e km/s  n_wind = %.1e cm^-3  P_wind = %.1e kb K cm^-3" %(v0/1e5, rho0/(mu*mp), P0/kb))

Edot_per_Vol = Edot / (4/3. * np.pi * r0**3) # source terms from SN
Mdot_per_Vol = Mdot / (4/3. * np.pi * r0**3) # source terms from SN

r_init = 100*pc

dv_dr0, drho_dr0, dP_dr0 = Hot_Wind_Evo(r0, np.r_[v0, rho0, P0])

dlogvdlogr   = dv_dr0 * r0/v0
dlogrhodlogr = drho_dr0 * r0/rho0
dlogPdlogr   = dP_dr0 * r0/P0
dlogr0       = 1e-8

v0_sub   = 10**(np.log10(v0) - dlogvdlogr * dlogr0)
rho0_sub = 10**(np.log10(rho0) - dlogrhodlogr * dlogr0)
P0_sub   = 10**(np.log10(P0) - dlogPdlogr * dlogr0)

sol = solve_ivp(Hot_Wind_Evo, [10**(np.log10(r0)-dlogr0),r_init], np.r_[v0_sub, rho0_sub, P0_sub], 
    events=[supersonic], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_init    = sol.t[-1]
v_init    = sol.y[0][-1]
rho_init  = sol.y[1][-1]
P_init    = sol.y[2][-1]
rhoZ_init = rho_init * Z_wind_initial

v0_sup   = 10**(np.log10(v0) + dlogvdlogr * dlogr0)
rho0_sup = 10**(np.log10(rho0) + dlogrhodlogr * dlogr0)
P0_sup   = 10**(np.log10(P0) + dlogPdlogr * dlogr0)

sol_sup = solve_ivp(Hot_Wind_Evo, [10**(np.log10(r0)+dlogr0),10**2*r0], np.r_[v0_sup, rho0_sup, P0_sup], 
    events=[supersonic], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_hot_only         = np.append(sol.t[::-1], sol_sup.t)
v_wind_hot_only    = np.append(sol.y[0][::-1], sol_sup.y[0])
rho_wind_hot_only  = np.append(sol.y[1][::-1], sol_sup.y[1])
P_wind_hot_only    = np.append(sol.y[2][::-1], sol_sup.y[2])

Mdot_wind_hot_only  = Omwind*r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only/(Msun/yr)
cs_wind_hot_only    = np.sqrt(gamma * P_wind_hot_only / rho_wind_hot_only)
T_wind_hot_only     = P_wind_hot_only/kb / (rho_wind_hot_only/(mu*mp))
K_wind_hot_only  = (P_wind_hot_only/kb) / (rho_wind_hot_only/(mu*mp))**gamma
Pdot_wind_hot_only = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only**2/(1e5*Msun/yr)
Pdot_wind_hot_only_P = Omwind * r_hot_only**2 * (rho_wind_hot_only * v_wind_hot_only**2 + P_wind_hot_only)/(1e5*Msun/yr)
Edot_wind_hot_only = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only * (0.5 * v_wind_hot_only**2 + 1.5 * cs_wind_hot_only**2)/(1e5**2*Msun/yr)

# cold cloud initial properties
T_cloud          = 1e4
log_eta_M_cold_tot = np.log10(eta_M_cold_tot)

# cold_cloud_injection_radial_power = -np.inf#6
# cold_cloud_injection_radial_extent = r0#1.33*r0
# cloud_radial_offest = 5e-2
cold_cloud_injection_radial_power = 6
cold_cloud_injection_radial_extent = 1.33*r0
cloud_radial_offest = 2e-2
irstart     = np.argmin(np.abs(r_hot_only-r0*(1.0+cloud_radial_offest)))
r_init      = r_hot_only[irstart]
v_init      = v_wind_hot_only[irstart]
rho_init    = rho_wind_hot_only[irstart]
P_init      = P_wind_hot_only[irstart]
M_cloud0         = 10**np.array([log_M_cloud0])*Msun
Z_cloud0         = np.ones_like(M_cloud0) * 1 * Z_solar #10**-1.0

# cold cloud total properties
v_cloud0        = np.ones_like(M_cloud0) * 10**1.5 * km/s # * v_init * v_factor #

Mdot_cold0      = eta_M_cold_tot * SFR
Ndot_cloud0     = Mdot_cold0 / M_cloud0

supersonic_initial_conditions = np.r_[v_init, rho_init, P_init, Z_wind_initial*rho_init, M_cloud0, v_cloud0, Z_cloud0]
sol = solve_ivp(Wind_Evo, [r_init, 1e2*r0], supersonic_initial_conditions, events=[supersonic,cloud_stop,cold_wind], dense_output=True,rtol=1e-10)
print(sol.message)
print(sol.t_events)

r           = sol.t
v_wind      = sol.y[0]
rho_wind    = sol.y[1]
P_wind      = sol.y[2]
rhoZ_wind   = sol.y[3]
M_cloud     = sol.y[4]
v_cloud     = sol.y[5]
Z_cloud     = sol.y[6]



cloud_Mdots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud / (Msun/yr))
Mdot_wind    = Omwind * r**2 * rho_wind * v_wind/(Msun/yr)
cs_wind = np.sqrt(gamma * P_wind / rho_wind)
T_wind  = P_wind/kb/(rho_wind/(mu*mp))
K_wind  = (P_wind/kb) / (rho_wind/(mu*mp))**gamma

Pdot_wind = Omwind * r**2 * rho_wind * v_wind**2/(1e5*Msun/yr)
Pdot_wind_P = Omwind * r**2 * (rho_wind * v_wind**2 + P_wind)/(1e5*Msun/yr)
cloud_Pdots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * v_cloud / (1e5 * Msun/yr))

Edot_wind = Omwind * r**2 * rho_wind * v_wind * (0.5 * v_wind**2 + 1.5 * cs_wind**2)/(1e5**2*Msun/yr)
cloud_Edots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * (0.5 * v_cloud**2 + 2.5 * kb * T_cloud/(mu*mp)) / (1e5**2 * Msun/yr))


fig,((axv,axZ,axd),(axM,axX,axP),(axMd,axEd,axK)) = plt.subplots(3,3,sharex=True,constrained_layout=True)
single_phase_color  = 'gray'
cs_linestyle        = ':'
cloud_linestyle     = '--'
axv.loglog(r_hot_only/kpc, v_wind_hot_only/1e5,                                    color = single_phase_color )
axv.loglog(r_hot_only/kpc, cs_wind_hot_only/1e5,                                   color = single_phase_color , ls = cs_linestyle)
axd.loglog(r_hot_only/kpc, rho_wind_hot_only/(mu*mp),                              color = single_phase_color )
axP.loglog(r_hot_only/kpc, P_wind_hot_only/kb,                                     color = single_phase_color )
axZ.loglog(r_hot_only/kpc, np.ones_like(rho_wind_hot_only)*Z_wind_initial/Z_solar, color = single_phase_color )
axK.loglog(r_hot_only/kpc, K_wind_hot_only,                                        color = single_phase_color )

axMd.loglog(r_hot_only/kpc, Mdot_wind_hot_only, color = single_phase_color )
axEd.loglog(r_hot_only/kpc, Edot_wind_hot_only, color = single_phase_color )

i_cloud = 2
cloud_colors = cmr.take_cmap_colors('cmr.infinity_s', 5, cmap_range=(0.1, 0.9), return_fmt='hex')

axv.loglog(r/kpc, v_wind/1e5,                                                        color = cloud_colors[i_cloud] )
axv.loglog(r/kpc, cs_wind/1e5,                                                       color = cloud_colors[i_cloud] ,ls = cs_linestyle )
axv.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, v_cloud)/1e5,        color = cloud_colors[i_cloud] ,ls = cloud_linestyle )
axd.loglog(r/kpc, rho_wind/(mu*mp),                                                  color = cloud_colors[i_cloud] )
axP.loglog(r/kpc, P_wind/kb,                                                         color = cloud_colors[i_cloud] )
axZ.loglog(r/kpc, rhoZ_wind / rho_wind / Z_solar,                              color = cloud_colors[i_cloud] )
axZ.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, Z_cloud)/Z_solar,    color = cloud_colors[i_cloud] ,ls = cloud_linestyle )
axK.loglog(r/kpc, K_wind,                                                            color = cloud_colors[i_cloud] )
axM.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)/Msun,       color = cloud_colors[i_cloud] )
if sol.status == 1:
    axM.scatter(r[-1]/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)[-1]/Msun,  color = cloud_colors[i_cloud] , marker='x', s=8)
axX.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_ksi(r, sol.y)), color = cloud_colors[i_cloud] )


axMd.loglog(r/kpc, Mdot_wind,                                                       color = cloud_colors[i_cloud] )
axMd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Mdots[0]),  color = cloud_colors[i_cloud] , ls = cloud_linestyle)


axEd.loglog(r/kpc, Edot_wind,                                                       color = cloud_colors[i_cloud] )
axEd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Edots[0]),      color = cloud_colors[i_cloud] , ls = cloud_linestyle)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cs_linestyle),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axv.legend(custom_lines, [r'$v_r$', r'$c_s$', r'$v_{\rm cl}$'],loc='lower center',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axZ.legend(custom_lines, [r'$Z$', r'$Z_{\rm cl}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axMd.legend(custom_lines, [r'${\rm hot}$', r'${\rm cold}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

axv.set_ylabel(r'$v \; [{\rm km/s}]$')
axd.set_ylabel(r'$n \; [{\rm cm}^{-3}]$')
axP.set_ylabel(r'$P \; [{\rm K cm}^{-3}]$')
axZ.set_ylabel(r'$Z \; [Z_\odot]$')
axK.set_ylabel(r'$K \; [{\rm K cm}^{2}]$')
axM.set_ylabel(r'$M_{\rm cl} \; [M_\odot]$')
axX.set_ylabel(r'$\xi = r_{\rm cl} / v_{\rm turb} t_{\rm cool}$')
axX.axhline(1, color='grey', lw=0.5, ls=':')

axMd.set_ylabel(r'$\dot{M} \; [M_\odot/{\rm yr}]$')
axEd.set_ylabel(r'$\dot{E} \; [{\rm km}^2/{\rm s}^2 \; M_\odot/{\rm yr}]$')

axMd.set_xlabel(r'$r\; [{\rm kpc}]$')
axEd.set_xlabel(r'$r\; [{\rm kpc}]$')
axK.set_xlabel(r'$r\; [{\rm kpc}]$')
plt.savefig('Single_Example_SFR'+str(int(SFR/(Msun/yr)))+'_etaE1_etaM'+str(np.round(eta_M,2))+'_etaMcold'+str(np.round(eta_M_cold_tot,2))+'_vcl3e1_Zcl1_distributed_injection_to_1+'+str(cloud_radial_offest)+'_'+str(np.round(cold_cloud_injection_radial_extent/r0,3))+'r0_power'+str(cold_cloud_injection_radial_power)+'_logMcl_'+str(log_M_cloud0)+'_vcirc'+str(np.round(v_circ0/1e5))+'_cooling_split_perfect.pdf',dpi=200,bbox_inches='tight')
plt.clf()










dv_dr       = np.zeros((len(r), 11))
drho_dr     = np.zeros((len(r), 11))
dP_dr       = np.zeros((len(r), 11))
dK_dr       = np.zeros((len(r), 5))
dM_cloud_dr = np.zeros((len(r), 3))
dv_cloud_dr = np.zeros((len(r), 4))

for i in range(len(r)):
    grads          = Gradient_Components(r[i], sol.y[:,i])
    dv_dr[i]       = np.array(grads[0])
    drho_dr[i]     = np.array(grads[1])
    dP_dr[i]       = np.array(grads[2])
    dK_dr[i]       = np.array(grads[3])
    dM_cloud_dr[i] = np.array(grads[4])
    dv_cloud_dr[i] = np.array(grads[5])

dlogv_dlogr       = (dv_dr.T * r/v_wind)
dlogrho_dlogr     = (drho_dr.T * r/rho_wind)
dlogP_dlogr       = (dP_dr.T * r/P_wind)
dlogK_dlogr       = (dK_dr.T * r/K_wind)
dlogM_cloud_dlogr = (dM_cloud_dr.T * r/M_cloud[0])
dlogv_cloud_dlogr = (dv_cloud_dr.T * r/v_cloud[0])

Mach            = v_wind / cs_wind
prefactor       = 1 / (1 - Mach**-2)
dlogv_dlogr     = (dlogv_dlogr/prefactor)
dlogrho_dlogr   = (dlogrho_dlogr/prefactor)
dlogP_dlogr     = (dlogP_dlogr/prefactor)


dencolors = ["#21c090","#157a5c"]
momcolors = ["#fd8c35", "#d95f02", "#8d3e01"]
enecolors = ["#b0afd5","#7e79b9","#524d93","#34315e"] 
basecolor = "#e7298a"

labels = ["total",
          "adiabatic",
          "gravity",
          "cloud growth",
          "cloud shredding",
          "momentum transfer",
          "kinetic thermalization",
          "thermal mixing",
          "cooling",
          "drag work",
          "drag force"]

fig, ((axv,axd,axP), (axK, axM, axV)) = plt.subplots(2,3,sharex=True, constrained_layout=True)

#total — velocity
line0, = axv.loglog(r/kpc,  dlogv_dlogr[0], color='k',               lw=3, label = labels[0])
axv.loglog(r/kpc, -dlogv_dlogr[0], color='k', dashes=[4,3], lw=3)
legend1 = axv.legend(handles=[line0], bbox_to_anchor=(0., 1.25, 1., .102), loc='upper left', ncol=1,  borderaxespad=0., fontsize=10, frameon=False) # mode="expand"
axv.add_artist(legend1)

#adiabatic
line1, = axv.loglog(r/kpc,  dlogv_dlogr[1], lw=2, color=basecolor, label = labels[1])
#density
line3, = axv.loglog(r/kpc,  dlogv_dlogr[3], lw=2, color=dencolors[0], label = labels[3])
axv.loglog(r/kpc,  -dlogv_dlogr[4], lw=2, color=dencolors[1], dashes=[4,3], zorder=10)
line4, = axv.loglog(r/kpc,  dlogv_dlogr[4], lw=2, color=dencolors[1], zorder=10, label = labels[4])
#momentum
axv.loglog(r/kpc,  -dlogv_dlogr[2], lw=2, color=momcolors[0], dashes=[4,3])
line2, = axv.loglog(r/kpc,  dlogv_dlogr[2], lw=2, color=momcolors[0], label = labels[2])
axv.loglog(r/kpc,  -dlogv_dlogr[5], lw=2, color=momcolors[1], dashes=[4,3])
line5, = axv.loglog(r/kpc,  dlogv_dlogr[5], lw=2, color=momcolors[1], label = labels[5])
axv.loglog(r/kpc,  -dlogv_dlogr[10],lw=2, color=momcolors[2], dashes=[4,3])
line10, = axv.loglog(r/kpc,  dlogv_dlogr[10],lw=2, color=momcolors[2], label = labels[10])
#energy
axv.loglog(r/kpc,  -dlogv_dlogr[6], lw=2, color=enecolors[0], dashes=[4,3])
line6, = axv.loglog(r/kpc,  dlogv_dlogr[6], lw=2, color=enecolors[0], label = labels[6])
line7, = axv.loglog(r/kpc,  dlogv_dlogr[7], lw=2, color=enecolors[1], label = labels[7])
line8, = axv.loglog(r/kpc,  dlogv_dlogr[8], lw=2, color=enecolors[2], label = labels[8])
line9, = axv.loglog(r/kpc, dlogv_dlogr[9], lw=2, color=enecolors[3], label = labels[9])
axv.loglog(r/kpc, -dlogv_dlogr[9], lw=2, color=enecolors[3], dashes=[4,3])
lines=[line1,line3,line4,line2,line5,line10,line6,line7,line8,line9]
axv.legend(handles=lines, bbox_to_anchor=(0.5, 1.02, 1., .102), loc='lower left', ncol=4,  borderaxespad=0., fontsize=10, frameon=False) # mode="expand"

#total — density
axd.loglog(r/kpc,  dlogrho_dlogr[0], color='k',               lw=3)
axd.loglog(r/kpc, -dlogrho_dlogr[0], color='k', dashes=[4,3], lw=3)
#adiabatic
axd.loglog(r/kpc, -dlogrho_dlogr[1], lw=2, color=basecolor, dashes=[4,3])
#density
axd.loglog(r/kpc, -dlogrho_dlogr[3], lw=2, color=dencolors[0], dashes=[4,3])
axd.loglog(r/kpc,  dlogrho_dlogr[4], lw=2, color=dencolors[1], zorder=10)
#momentum
axd.loglog(r/kpc, dlogrho_dlogr[2], lw=2, color=momcolors[0])
axd.loglog(r/kpc, dlogrho_dlogr[5], lw=2, color=momcolors[1])
axd.loglog(r/kpc, dlogrho_dlogr[10],lw=2, color=momcolors[2])
#energy
axd.loglog(r/kpc,  dlogrho_dlogr[6], lw=2, color=enecolors[0])
axd.loglog(r/kpc, -dlogrho_dlogr[7], lw=2, color=enecolors[1], dashes=[4,3])
axd.loglog(r/kpc, -dlogrho_dlogr[8], lw=2, color=enecolors[2], dashes=[4,3])
axd.loglog(r/kpc,  dlogrho_dlogr[9], lw=2, color=enecolors[3])

#total — pressure
axP.loglog(r/kpc,  dlogP_dlogr[0], color='k', lw=3)
axP.loglog(r/kpc, -dlogP_dlogr[0], color='k', dashes=[4,3], lw=3)
#adiabatic
axP.loglog(r/kpc, -dlogP_dlogr[1], lw=2, color=basecolor, dashes=[4,3])
#density
axP.loglog(r/kpc, -dlogP_dlogr[3], lw=2, color=dencolors[0], dashes=[4,3])
axP.loglog(r/kpc,  dlogP_dlogr[4], lw=2, color=dencolors[1])
#momentum
axP.loglog(r/kpc, dlogP_dlogr[2], lw=2, color=momcolors[0])
axP.loglog(r/kpc, dlogP_dlogr[5], lw=2, color=momcolors[1])
axP.loglog(r/kpc, dlogP_dlogr[10],lw=2, color=momcolors[2])
#energy
axP.loglog(r/kpc,  dlogP_dlogr[6], lw=2, color=enecolors[0])
axP.loglog(r/kpc, -dlogP_dlogr[7], lw=2, color=enecolors[1], dashes=[4,3])
axP.loglog(r/kpc, -dlogP_dlogr[8], lw=2, color=enecolors[2], dashes=[4,3])
axP.loglog(r/kpc,  dlogP_dlogr[9], lw=2, color=enecolors[3])

#total — entropy
axK.loglog(r/kpc,  dlogK_dlogr[0], color='k',               lw=3)
axK.loglog(r/kpc, -dlogK_dlogr[0], color='k', dashes=[4,3], lw=3)
axK.loglog(r/kpc,  dlogK_dlogr[1], lw=2, color=enecolors[0])
axK.loglog(r/kpc, -dlogK_dlogr[2], lw=2, color=enecolors[1], dashes=[4,3])
axK.loglog(r/kpc, -dlogK_dlogr[3], lw=2, color=enecolors[2], dashes=[4,3])
axK.loglog(r/kpc,  dlogK_dlogr[4], lw=2, color=enecolors[3])

axM.loglog(r/kpc,  dlogM_cloud_dlogr[0], color='k', lw=3)
axM.loglog(r/kpc, -dlogM_cloud_dlogr[0], color='k', dashes=[4,3], lw=3)
axM.loglog(r/kpc,  dlogM_cloud_dlogr[1], lw=2, color=dencolors[0])
axM.loglog(r/kpc, -dlogM_cloud_dlogr[2], lw=2, color=dencolors[1], dashes=[4,3])

axV.loglog(r/kpc,  dlogv_cloud_dlogr[0], color='k', lw=3)
axV.loglog(r/kpc, -dlogv_cloud_dlogr[0], dashes=[4,3], color='k', lw=3)
axV.loglog(r/kpc,  dlogv_cloud_dlogr[1], lw=2, color=momcolors[1])
axV.loglog(r/kpc,  dlogv_cloud_dlogr[2], lw=2, color=momcolors[2])
axV.loglog(r/kpc, -dlogv_cloud_dlogr[3], lw=2, color=momcolors[0],dashes=[4,3])

axv.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log v}{d \log r}$}    \fontsize{10pt}{3em}\selectfont{}{$(1 {-} \mathcal{M}^{-2})$}", size=12)
axd.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log \rho}{d \log r}$} \fontsize{10pt}{3em}\selectfont{}{$(1 {-} \mathcal{M}^{-2})$}", size=12)
axP.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log P}{d \log r}$}    \fontsize{10pt}{3em}\selectfont{}{$(1 {-} \mathcal{M}^{-2})$}", size=12)
axK.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log K}{d \log r}$}", size=12)
axM.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log M_{\rm cl}}{d \log r}$}", size=12)
axV.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log v_{\rm cl}}{d \log r}$}", size=12)
axK.set_xlabel(r'$r \; [{\rm kpc}]$')
axM.set_xlabel(r'$r \; [{\rm kpc}]$')
axV.set_xlabel(r'$r \; [{\rm kpc}]$')




axv.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
axv.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
axd.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
axd.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))

axM.text(0.5, 0.95,
    r"${\rm SFR} = 20 M_\odot / {\rm yr}$" + "\n" +
    r"$\eta_{\rm M} = 0.1$" + "\n" +
    r"$\eta_{\rm M, cold} = 0.2$" + "\n" + 
    r"$M_{\rm cloud} = 10^3 M_\odot$", 
    ha='left', va='top', color = 'k', fontsize=8, transform=axM.transAxes)

fig.subplots_adjust(wspace=0.4,hspace=0.1)
fig.set_size_inches(9.5,5)
plt.savefig('test_Single_Example_color_coded_gradients_SFR'+str(int(SFR/(Msun/yr)))+'_etaE1_etaM'+str(np.round(eta_M,2))+'_etaMcold'+str(np.round(eta_M_cold_tot,2))+'_vcl3e1_Zcl1_distributed_injection_to_1+'+str(cloud_radial_offest)+'_'+str(np.round(cold_cloud_injection_radial_extent/r0,3))+'r0_power'+str(cold_cloud_injection_radial_power)+'_logMcl_'+str(log_M_cloud0)+'_vcirc'+str(np.round(v_circ0/1e5))+'_cooling_split_perfect.pdf',dpi=200,bbox_inches='tight')
plt.clf()











###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>






SFR = 20 * Msun/yr
eta_M              = 0.5
eta_M_cold_tot     = 0.5
log_M_cloud0       = 3

### METALLICITY
Z_solar = 0.02

# feedback and SF props
E_SN  = 1e51
mstar = 100*Msun
M_cloud_min = 1e-2*Msun

## model choices
CoolingAreaChiPower         =  0.5 
ColdTurbulenceChiPower      = -0.5 
TurbulentVelocityChiPower   =  0.0 
geometric_factor            = 1.0
Mdot_coefficient            = 1.0/3.0
Cooling_Factor              = 1.0
drag_coeff                  = 0.5
f_turb0                     = 10**-1.0
v_circ0                     = 150e5 # gravitational potential assuming isothermal potential

r0                 = 300*pc
Z_wind_initial     = 2.0 * Z_solar
half_opening_angle = np.pi/2
Omwind             = 4*np.pi*(1.0 - np.cos(half_opening_angle))
eta_E              = 1

Mdot        = eta_M * SFR
Edot        = eta_E * (E_SN/mstar) * SFR

# properties at r0 if no clouds + gravity
epsilon     = 1e-5
Mach0       = 1.0 + epsilon
v0          = np.sqrt(Edot/Mdot)*(1/((gamma-1)*Mach0) + 1/2.)**(-1/2.)
rho0        = Mdot/(Omwind*r0**2 * v0)
P0          = rho0*v0**2 / Mach0**2 / gamma
rhoZ0       = rho0 * Z_wind_initial
print( "v_wind = %.1e km/s  n_wind = %.1e cm^-3  P_wind = %.1e kb K cm^-3" %(v0/1e5, rho0/(mu*mp), P0/kb))

Edot_per_Vol = Edot / (4/3. * np.pi * r0**3) # source terms from SN
Mdot_per_Vol = Mdot / (4/3. * np.pi * r0**3) # source terms from SN

r_init = 100*pc

dv_dr0, drho_dr0, dP_dr0 = Hot_Wind_Evo(r0, np.r_[v0, rho0, P0])

dlogvdlogr   = dv_dr0 * r0/v0
dlogrhodlogr = drho_dr0 * r0/rho0
dlogPdlogr   = dP_dr0 * r0/P0
dlogr0       = 1e-8

v0_sub   = 10**(np.log10(v0) - dlogvdlogr * dlogr0)
rho0_sub = 10**(np.log10(rho0) - dlogrhodlogr * dlogr0)
P0_sub   = 10**(np.log10(P0) - dlogPdlogr * dlogr0)

sol = solve_ivp(Hot_Wind_Evo, [10**(np.log10(r0)-dlogr0),r_init], np.r_[v0_sub, rho0_sub, P0_sub], 
    events=[supersonic], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_init    = sol.t[-1]
v_init    = sol.y[0][-1]
rho_init  = sol.y[1][-1]
P_init    = sol.y[2][-1]
rhoZ_init = rho_init * Z_wind_initial

v0_sup   = 10**(np.log10(v0) + dlogvdlogr * dlogr0)
rho0_sup = 10**(np.log10(rho0) + dlogrhodlogr * dlogr0)
P0_sup   = 10**(np.log10(P0) + dlogPdlogr * dlogr0)

sol_sup = solve_ivp(Hot_Wind_Evo, [10**(np.log10(r0)+dlogr0),10**2*r0], np.r_[v0_sup, rho0_sup, P0_sup], 
    events=[supersonic], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_hot_only         = np.append(sol.t[::-1], sol_sup.t)
v_wind_hot_only    = np.append(sol.y[0][::-1], sol_sup.y[0])
rho_wind_hot_only  = np.append(sol.y[1][::-1], sol_sup.y[1])
P_wind_hot_only    = np.append(sol.y[2][::-1], sol_sup.y[2])

Mdot_wind_hot_only  = Omwind*r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only/(Msun/yr)
cs_wind_hot_only    = np.sqrt(gamma * P_wind_hot_only / rho_wind_hot_only)
T_wind_hot_only     = P_wind_hot_only/kb / (rho_wind_hot_only/(mu*mp))
K_wind_hot_only  = (P_wind_hot_only/kb) / (rho_wind_hot_only/(mu*mp))**gamma
Pdot_wind_hot_only = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only**2/(1e5*Msun/yr)
Pdot_wind_hot_only_P = Omwind * r_hot_only**2 * (rho_wind_hot_only * v_wind_hot_only**2 + P_wind_hot_only)/(1e5*Msun/yr)
Edot_wind_hot_only = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only * (0.5 * v_wind_hot_only**2 + 1.5 * cs_wind_hot_only**2)/(1e5**2*Msun/yr)

# cold cloud initial properties
T_cloud          = 1e4
log_eta_M_cold_tot = np.log10(eta_M_cold_tot)

# cold_cloud_injection_radial_power = -np.inf#6
# cold_cloud_injection_radial_extent = r0#1.33*r0
# cloud_radial_offest = 5e-2
cold_cloud_injection_radial_power = 6
cold_cloud_injection_radial_extent = 1.33*r0
cloud_radial_offest = 2e-2
irstart     = np.argmin(np.abs(r_hot_only-r0*(1.0+cloud_radial_offest)))
r_init      = r_hot_only[irstart]
v_init      = v_wind_hot_only[irstart]
rho_init    = rho_wind_hot_only[irstart]
P_init      = P_wind_hot_only[irstart]
M_cloud0         = 10**np.array([log_M_cloud0])*Msun
Z_cloud0         = np.ones_like(M_cloud0) * 1 * Z_solar #10**-1.0

# cold cloud total properties
v_cloud0        = np.ones_like(M_cloud0) * 10**1.5 * km/s # * v_init * v_factor #

Mdot_cold0      = eta_M_cold_tot * SFR
Ndot_cloud0     = Mdot_cold0 / M_cloud0

supersonic_initial_conditions = np.r_[v_init, rho_init, P_init, Z_wind_initial*rho_init, M_cloud0, v_cloud0, Z_cloud0]
sol = solve_ivp(Wind_Evo, [r_init, 1e2*r0], supersonic_initial_conditions, events=[supersonic,cloud_stop,cold_wind], dense_output=True,rtol=1e-10)
print(sol.message)
print(sol.t_events)

r           = sol.t
v_wind      = sol.y[0]
rho_wind    = sol.y[1]
P_wind      = sol.y[2]
rhoZ_wind   = sol.y[3]
M_cloud     = sol.y[4]
v_cloud     = sol.y[5]
Z_cloud     = sol.y[6]



cloud_Mdots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud / (Msun/yr))
Mdot_wind    = Omwind * r**2 * rho_wind * v_wind/(Msun/yr)
cs_wind = np.sqrt(gamma * P_wind / rho_wind)
T_wind  = P_wind/kb/(rho_wind/(mu*mp))
K_wind  = (P_wind/kb) / (rho_wind/(mu*mp))**gamma

Pdot_wind = Omwind * r**2 * rho_wind * v_wind**2/(1e5*Msun/yr)
Pdot_wind_P = Omwind * r**2 * (rho_wind * v_wind**2 + P_wind)/(1e5*Msun/yr)
cloud_Pdots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * v_cloud / (1e5 * Msun/yr))

Edot_wind = Omwind * r**2 * rho_wind * v_wind * (0.5 * v_wind**2 + 1.5 * cs_wind**2)/(1e5**2*Msun/yr)
cloud_Edots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * (0.5 * v_cloud**2 + 2.5 * kb * T_cloud/(mu*mp)) / (1e5**2 * Msun/yr))


fig,((axv,axZ,axd),(axM,axX,axP),(axMd,axEd,axK)) = plt.subplots(3,3,sharex=True,constrained_layout=True)
single_phase_color  = 'gray'
cs_linestyle        = ':'
cloud_linestyle     = '--'
axv.loglog(r_hot_only/kpc, v_wind_hot_only/1e5,                                    color = single_phase_color )
axv.loglog(r_hot_only/kpc, cs_wind_hot_only/1e5,                                   color = single_phase_color , ls = cs_linestyle)
axd.loglog(r_hot_only/kpc, rho_wind_hot_only/(mu*mp),                              color = single_phase_color )
axP.loglog(r_hot_only/kpc, P_wind_hot_only/kb,                                     color = single_phase_color )
axZ.loglog(r_hot_only/kpc, np.ones_like(rho_wind_hot_only)*Z_wind_initial/Z_solar, color = single_phase_color )
axK.loglog(r_hot_only/kpc, K_wind_hot_only,                                        color = single_phase_color )

axMd.loglog(r_hot_only/kpc, Mdot_wind_hot_only, color = single_phase_color )
axEd.loglog(r_hot_only/kpc, Edot_wind_hot_only, color = single_phase_color )

i_cloud = 2
cloud_colors = cmr.take_cmap_colors('cmr.infinity_s', 5, cmap_range=(0.1, 0.9), return_fmt='hex')

axv.loglog(r/kpc, v_wind/1e5,                                                        color = cloud_colors[i_cloud] )
axv.loglog(r/kpc, cs_wind/1e5,                                                       color = cloud_colors[i_cloud] ,ls = cs_linestyle )
axv.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, v_cloud)/1e5,        color = cloud_colors[i_cloud] ,ls = cloud_linestyle )
axd.loglog(r/kpc, rho_wind/(mu*mp),                                                  color = cloud_colors[i_cloud] )
axP.loglog(r/kpc, P_wind/kb,                                                         color = cloud_colors[i_cloud] )
axZ.loglog(r/kpc, rhoZ_wind / rho_wind / Z_solar,                              color = cloud_colors[i_cloud] )
axZ.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, Z_cloud)/Z_solar,    color = cloud_colors[i_cloud] ,ls = cloud_linestyle )
axK.loglog(r/kpc, K_wind,                                                            color = cloud_colors[i_cloud] )
axM.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)/Msun,       color = cloud_colors[i_cloud] )
if sol.status == 1:
    axM.scatter(r[-1]/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)[-1]/Msun,  color = cloud_colors[i_cloud] , marker='x', s=8)
axX.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_ksi(r, sol.y)), color = cloud_colors[i_cloud] )


axMd.loglog(r/kpc, Mdot_wind,                                                       color = cloud_colors[i_cloud] )
axMd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Mdots[0]),  color = cloud_colors[i_cloud] , ls = cloud_linestyle)


axEd.loglog(r/kpc, Edot_wind,                                                       color = cloud_colors[i_cloud] )
axEd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Edots[0]),      color = cloud_colors[i_cloud] , ls = cloud_linestyle)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cs_linestyle),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axv.legend(custom_lines, [r'$v_r$', r'$c_s$', r'$v_{\rm cl}$'],loc='lower center',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axZ.legend(custom_lines, [r'$Z$', r'$Z_{\rm cl}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axMd.legend(custom_lines, [r'${\rm hot}$', r'${\rm cold}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

axv.set_ylabel(r'$v \; [{\rm km/s}]$')
axd.set_ylabel(r'$n \; [{\rm cm}^{-3}]$')
axP.set_ylabel(r'$P \; [{\rm K cm}^{-3}]$')
axZ.set_ylabel(r'$Z \; [Z_\odot]$')
axK.set_ylabel(r'$K \; [{\rm K cm}^{2}]$')
axM.set_ylabel(r'$M_{\rm cl} \; [M_\odot]$')
axX.set_ylabel(r'$\xi = r_{\rm cl} / v_{\rm turb} t_{\rm cool}$')
axX.axhline(1, color='grey', lw=0.5, ls=':')

axMd.set_ylabel(r'$\dot{M} \; [M_\odot/{\rm yr}]$')
axEd.set_ylabel(r'$\dot{E} \; [{\rm km}^2/{\rm s}^2 \; M_\odot/{\rm yr}]$')

axMd.set_xlabel(r'$r\; [{\rm kpc}]$')
axEd.set_xlabel(r'$r\; [{\rm kpc}]$')
axK.set_xlabel(r'$r\; [{\rm kpc}]$')
plt.savefig('Single_Example_SFR'+str(int(SFR/(Msun/yr)))+'_etaE1_etaM'+str(np.round(eta_M,2))+'_etaMcold'+str(np.round(eta_M_cold_tot,2))+'_vcl3e1_Zcl1_distributed_injection_to_1+'+str(cloud_radial_offest)+'_'+str(np.round(cold_cloud_injection_radial_extent/r0,3))+'r0_power'+str(cold_cloud_injection_radial_power)+'_logMcl_'+str(log_M_cloud0)+'_vcirc'+str(np.round(v_circ0/1e5))+'_cooling_split_perfect.pdf',dpi=200,bbox_inches='tight')
plt.clf()










dv_dr       = np.zeros((len(r), 11))
drho_dr     = np.zeros((len(r), 11))
dP_dr       = np.zeros((len(r), 11))
dK_dr       = np.zeros((len(r), 5))
dM_cloud_dr = np.zeros((len(r), 3))
dv_cloud_dr = np.zeros((len(r), 4))

for i in range(len(r)):
    grads          = Gradient_Components(r[i], sol.y[:,i])
    dv_dr[i]       = np.array(grads[0])
    drho_dr[i]     = np.array(grads[1])
    dP_dr[i]       = np.array(grads[2])
    dK_dr[i]       = np.array(grads[3])
    dM_cloud_dr[i] = np.array(grads[4])
    dv_cloud_dr[i] = np.array(grads[5])

dlogv_dlogr       = (dv_dr.T * r/v_wind)
dlogrho_dlogr     = (drho_dr.T * r/rho_wind)
dlogP_dlogr       = (dP_dr.T * r/P_wind)
dlogK_dlogr       = (dK_dr.T * r/K_wind)
dlogM_cloud_dlogr = (dM_cloud_dr.T * r/M_cloud[0])
dlogv_cloud_dlogr = (dv_cloud_dr.T * r/v_cloud[0])

Mach            = v_wind / cs_wind
prefactor       = 1 / (1 - Mach**-2)
dlogv_dlogr     = (dlogv_dlogr/prefactor)
dlogrho_dlogr   = (dlogrho_dlogr/prefactor)
dlogP_dlogr     = (dlogP_dlogr/prefactor)


dencolors = ["#21c090","#157a5c"]
momcolors = ["#fd8c35", "#d95f02", "#8d3e01"]
enecolors = ["#b0afd5","#7e79b9","#524d93","#34315e"] 
basecolor = "#e7298a"

labels = ["total",
          "adiabatic",
          "gravity",
          "cloud growth",
          "cloud shredding",
          "momentum transfer",
          "kinetic thermalization",
          "thermal mixing",
          "cooling",
          "drag work",
          "drag force"]

fig, ((axv,axd,axP), (axK, axM, axV)) = plt.subplots(2,3,sharex=True, constrained_layout=True)

#total — velocity
line0, = axv.loglog(r/kpc,  dlogv_dlogr[0], color='k',               lw=3, label = labels[0])
axv.loglog(r/kpc, -dlogv_dlogr[0], color='k', dashes=[4,3], lw=3)
legend1 = axv.legend(handles=[line0], bbox_to_anchor=(0., 1.25, 1., .102), loc='upper left', ncol=1,  borderaxespad=0., fontsize=10, frameon=False) # mode="expand"
axv.add_artist(legend1)

#adiabatic
line1, = axv.loglog(r/kpc,  dlogv_dlogr[1], lw=2, color=basecolor, label = labels[1])
#density
line3, = axv.loglog(r/kpc,  dlogv_dlogr[3], lw=2, color=dencolors[0], label = labels[3])
axv.loglog(r/kpc,  -dlogv_dlogr[4], lw=2, color=dencolors[1], dashes=[4,3], zorder=10)
line4, = axv.loglog(r/kpc,  dlogv_dlogr[4], lw=2, color=dencolors[1], zorder=10, label = labels[4])
#momentum
axv.loglog(r/kpc,  -dlogv_dlogr[2], lw=2, color=momcolors[0], dashes=[4,3])
line2, = axv.loglog(r/kpc,  dlogv_dlogr[2], lw=2, color=momcolors[0], label = labels[2])
axv.loglog(r/kpc,  -dlogv_dlogr[5], lw=2, color=momcolors[1], dashes=[4,3])
line5, = axv.loglog(r/kpc,  dlogv_dlogr[5], lw=2, color=momcolors[1], label = labels[5])
axv.loglog(r/kpc,  -dlogv_dlogr[10],lw=2, color=momcolors[2], dashes=[4,3])
line10, = axv.loglog(r/kpc,  dlogv_dlogr[10],lw=2, color=momcolors[2], label = labels[10])
#energy
axv.loglog(r/kpc,  -dlogv_dlogr[6], lw=2, color=enecolors[0], dashes=[4,3])
line6, = axv.loglog(r/kpc,  dlogv_dlogr[6], lw=2, color=enecolors[0], label = labels[6])
line7, = axv.loglog(r/kpc,  dlogv_dlogr[7], lw=2, color=enecolors[1], label = labels[7])
line8, = axv.loglog(r/kpc,  dlogv_dlogr[8], lw=2, color=enecolors[2], label = labels[8])
line9, = axv.loglog(r/kpc, dlogv_dlogr[9], lw=2, color=enecolors[3], label = labels[9])
axv.loglog(r/kpc, -dlogv_dlogr[9], lw=2, color=enecolors[3], dashes=[4,3])
lines=[line1,line3,line4,line2,line5,line10,line6,line7,line8,line9]
axv.legend(handles=lines, bbox_to_anchor=(0.5, 1.02, 1., .102), loc='lower left', ncol=4,  borderaxespad=0., fontsize=10, frameon=False) # mode="expand"

#total — density
axd.loglog(r/kpc,  dlogrho_dlogr[0], color='k',               lw=3)
axd.loglog(r/kpc, -dlogrho_dlogr[0], color='k', dashes=[4,3], lw=3)
#adiabatic
axd.loglog(r/kpc, -dlogrho_dlogr[1], lw=2, color=basecolor, dashes=[4,3])
#density
axd.loglog(r/kpc, -dlogrho_dlogr[3], lw=2, color=dencolors[0], dashes=[4,3])
axd.loglog(r/kpc,  dlogrho_dlogr[4], lw=2, color=dencolors[1], zorder=10)
#momentum
axd.loglog(r/kpc, dlogrho_dlogr[2], lw=2, color=momcolors[0])
axd.loglog(r/kpc, dlogrho_dlogr[5], lw=2, color=momcolors[1])
axd.loglog(r/kpc, dlogrho_dlogr[10],lw=2, color=momcolors[2])
#energy
axd.loglog(r/kpc,  dlogrho_dlogr[6], lw=2, color=enecolors[0])
axd.loglog(r/kpc, -dlogrho_dlogr[7], lw=2, color=enecolors[1], dashes=[4,3])
axd.loglog(r/kpc, -dlogrho_dlogr[8], lw=2, color=enecolors[2], dashes=[4,3])
axd.loglog(r/kpc,  dlogrho_dlogr[9], lw=2, color=enecolors[3])

#total — pressure
axP.loglog(r/kpc,  dlogP_dlogr[0], color='k', lw=3)
axP.loglog(r/kpc, -dlogP_dlogr[0], color='k', dashes=[4,3], lw=3)
#adiabatic
axP.loglog(r/kpc, -dlogP_dlogr[1], lw=2, color=basecolor, dashes=[4,3])
#density
axP.loglog(r/kpc, -dlogP_dlogr[3], lw=2, color=dencolors[0], dashes=[4,3])
axP.loglog(r/kpc,  dlogP_dlogr[4], lw=2, color=dencolors[1])
#momentum
axP.loglog(r/kpc, dlogP_dlogr[2], lw=2, color=momcolors[0])
axP.loglog(r/kpc, dlogP_dlogr[5], lw=2, color=momcolors[1])
axP.loglog(r/kpc, dlogP_dlogr[10],lw=2, color=momcolors[2])
#energy
axP.loglog(r/kpc,  dlogP_dlogr[6], lw=2, color=enecolors[0])
axP.loglog(r/kpc, -dlogP_dlogr[7], lw=2, color=enecolors[1], dashes=[4,3])
axP.loglog(r/kpc, -dlogP_dlogr[8], lw=2, color=enecolors[2], dashes=[4,3])
axP.loglog(r/kpc,  dlogP_dlogr[9], lw=2, color=enecolors[3])

#total — entropy
axK.loglog(r/kpc,  dlogK_dlogr[0], color='k',               lw=3)
axK.loglog(r/kpc, -dlogK_dlogr[0], color='k', dashes=[4,3], lw=3)
axK.loglog(r/kpc,  dlogK_dlogr[1], lw=2, color=enecolors[0])
axK.loglog(r/kpc, -dlogK_dlogr[2], lw=2, color=enecolors[1], dashes=[4,3])
axK.loglog(r/kpc, -dlogK_dlogr[3], lw=2, color=enecolors[2], dashes=[4,3])
axK.loglog(r/kpc,  dlogK_dlogr[4], lw=2, color=enecolors[3])

axM.loglog(r/kpc,  dlogM_cloud_dlogr[0], color='k', lw=3)
axM.loglog(r/kpc, -dlogM_cloud_dlogr[0], color='k', dashes=[4,3], lw=3)
axM.loglog(r/kpc,  dlogM_cloud_dlogr[1], lw=2, color=dencolors[0])
axM.loglog(r/kpc, -dlogM_cloud_dlogr[2], lw=2, color=dencolors[1], dashes=[4,3])

axV.loglog(r/kpc,  dlogv_cloud_dlogr[0], color='k', lw=3)
axV.loglog(r/kpc, -dlogv_cloud_dlogr[0], dashes=[4,3], color='k', lw=3)
axV.loglog(r/kpc,  dlogv_cloud_dlogr[1], lw=2, color=momcolors[1])
axV.loglog(r/kpc,  dlogv_cloud_dlogr[2], lw=2, color=momcolors[2])
axV.loglog(r/kpc, -dlogv_cloud_dlogr[3], lw=2, color=momcolors[0],dashes=[4,3])

axv.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log v}{d \log r}$}    \fontsize{10pt}{3em}\selectfont{}{$(1 {-} \mathcal{M}^{-2})$}", size=12)
axd.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log \rho}{d \log r}$} \fontsize{10pt}{3em}\selectfont{}{$(1 {-} \mathcal{M}^{-2})$}", size=12)
axP.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log P}{d \log r}$}    \fontsize{10pt}{3em}\selectfont{}{$(1 {-} \mathcal{M}^{-2})$}", size=12)
axK.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log K}{d \log r}$}", size=12)
axM.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log M_{\rm cl}}{d \log r}$}", size=12)
axV.set_ylabel(r"\fontsize{14pt}{3em}\selectfont{}{$\frac{d \log v_{\rm cl}}{d \log r}$}", size=12)
axK.set_xlabel(r'$r \; [{\rm kpc}]$')
axM.set_xlabel(r'$r \; [{\rm kpc}]$')
axV.set_xlabel(r'$r \; [{\rm kpc}]$')



axv.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
axv.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
axd.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0])
axd.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))


axv.set_ylim(bottom=2e-5)
axd.set_ylim(bottom=2e-4)
axP.set_ylim(bottom=2e-3)
axM.set_ylim(bottom=9e-3)
axV.set_ylim(bottom=9e-3)

axM.text(0.5, 0.95,
    r"${\rm SFR} = 20 M_\odot / {\rm yr}$" + "\n" +
    r"$\eta_{\rm M} = 0.5$" + "\n" +
    r"$\eta_{\rm M, cold} = 0.5$" + "\n" + 
    r"$M_{\rm cloud} = 10^3 M_\odot$", 
    ha='left', va='top', color = 'k', fontsize=8, transform=axM.transAxes)

fig.subplots_adjust(wspace=0.4,hspace=0.1)
fig.set_size_inches(9.5,5)
plt.savefig('test_Single_Example_color_coded_gradients_SFR'+str(int(SFR/(Msun/yr)))+'_etaE1_etaM'+str(np.round(eta_M,2))+'_etaMcold'+str(np.round(eta_M_cold_tot,2))+'_vcl3e1_Zcl1_distributed_injection_to_1+'+str(cloud_radial_offest)+'_'+str(np.round(cold_cloud_injection_radial_extent/r0,3))+'r0_power'+str(cold_cloud_injection_radial_power)+'_logMcl_'+str(log_M_cloud0)+'_vcirc'+str(np.round(v_circ0/1e5))+'_cooling_split_perfect.pdf',dpi=200,bbox_inches='tight')
plt.clf()














#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#






#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#















#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
#()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()##()#
































###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>






SFR = 20 * Msun/yr

eta_M              = 0.1
eta_M_cold_tot     = 0.2


### METALLICITY
Z_solar = 0.02

# feedback and SF props
E_SN  = 1e51
mstar = 100*Msun
M_cloud_min = 1e-2*Msun

## model choices
CoolingAreaChiPower         =  0.5 
ColdTurbulenceChiPower      = -0.5 
TurbulentVelocityChiPower   =  0.0 
geometric_factor            = 1.0
Mdot_coefficient            = 1.0/3.0
Cooling_Factor              = 1.0
drag_coeff                  = 0.5
f_turb0                     = 10**-1.0
v_circ0                     = 150e5 # gravitational potential assuming isothermal potential


eta_E              = 1
r0                 = 300*pc
Z_wind_initial     = 2.0 * Z_solar
half_opening_angle = np.pi/2
Omwind             = 4*np.pi*(1.0 - np.cos(half_opening_angle))

Mdot        = eta_M * SFR
Edot        = eta_E * (E_SN/mstar) * SFR

# properties at r0 if no clouds + gravity
epsilon     = 1e-5
Mach0       = 1.0 + epsilon
v0          = np.sqrt(Edot/Mdot)*(1/((gamma-1)*Mach0) + 1/2.)**(-1/2.)
rho0        = Mdot/(Omwind*r0**2 * v0)
P0          = rho0*v0**2 / Mach0**2 / gamma
rhoZ0       = rho0 * Z_wind_initial
print( "v_wind = %.1e km/s  n_wind = %.1e cm^-3  P_wind = %.1e kb K cm^-3" %(v0/1e5, rho0/(mu*mp), P0/kb))

Edot_per_Vol = Edot / (4/3. * np.pi * r0**3) # source terms from SN
Mdot_per_Vol = Mdot / (4/3. * np.pi * r0**3) # source terms from SN

r_init = 100*pc

dv_dr0, drho_dr0, dP_dr0 = Hot_Wind_Evo(r0, np.r_[v0, rho0, P0])

dlogvdlogr   = dv_dr0 * r0/v0
dlogrhodlogr = drho_dr0 * r0/rho0
dlogPdlogr   = dP_dr0 * r0/P0
dlogr0       = 1e-8

v0_sub   = 10**(np.log10(v0) - dlogvdlogr * dlogr0)
rho0_sub = 10**(np.log10(rho0) - dlogrhodlogr * dlogr0)
P0_sub   = 10**(np.log10(P0) - dlogPdlogr * dlogr0)

sol = solve_ivp(Hot_Wind_Evo, [10**(np.log10(r0)-dlogr0),r_init], np.r_[v0_sub, rho0_sub, P0_sub], 
    events=[supersonic,cold_wind], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_init    = sol.t[-1]
v_init    = sol.y[0][-1]
rho_init  = sol.y[1][-1]
P_init    = sol.y[2][-1]
rhoZ_init = rho_init * Z_wind_initial

v0_sup   = 10**(np.log10(v0) + dlogvdlogr * dlogr0)
rho0_sup = 10**(np.log10(rho0) + dlogrhodlogr * dlogr0)
P0_sup   = 10**(np.log10(P0) + dlogPdlogr * dlogr0)

sol_sup = solve_ivp(Hot_Wind_Evo, [10**(np.log10(r0)+dlogr0),10**2*r0], np.r_[v0_sup, rho0_sup, P0_sup], 
    events=[supersonic,cold_wind], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_hot_only         = np.append(sol.t[::-1], sol_sup.t)
v_wind_hot_only    = np.append(sol.y[0][::-1], sol_sup.y[0])
rho_wind_hot_only  = np.append(sol.y[1][::-1], sol_sup.y[1])
P_wind_hot_only    = np.append(sol.y[2][::-1], sol_sup.y[2])

Mdot_wind_hot_only  = Omwind*r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only/(Msun/yr)
cs_wind_hot_only    = np.sqrt(gamma * P_wind_hot_only / rho_wind_hot_only)
T_wind_hot_only     = P_wind_hot_only/kb / (rho_wind_hot_only/(mu*mp))
K_wind_hot_only  = (P_wind_hot_only/kb) / (rho_wind_hot_only/(mu*mp))**gamma
Pdot_wind_hot_only = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only**2/(1e5*Msun/yr)
Pdot_wind_hot_only_P = Omwind * r_hot_only**2 * (rho_wind_hot_only * v_wind_hot_only**2 + P_wind_hot_only)/(1e5*Msun/yr)
Edot_wind_hot_only = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only * (0.5 * v_wind_hot_only**2 + 1.5 * cs_wind_hot_only**2)/(1e5**2*Msun/yr)


# cold cloud initial properties
T_cloud             = 1e4
log_cloud_masses    = np.array([2,3,4,5,6])
N_cloud_masses      = len(log_cloud_masses)
cloud_colors        = cmr.take_cmap_colors('cmr.infinity_s', N_cloud_masses, cmap_range=(0.1, 0.9), return_fmt='hex')

fig,((axv,axZ,axd),(axM,axX,axP),(axMd,axEd,axK)) = plt.subplots(3,3,sharex=True,constrained_layout=True)
single_phase_color  = 'gray'
cs_linestyle        = ':'
cloud_linestyle     = '--'
axv.loglog(r_hot_only/kpc, v_wind_hot_only/1e5,                                    color = single_phase_color )
axv.loglog(r_hot_only/kpc, cs_wind_hot_only/1e5,                                   color = single_phase_color , ls = cs_linestyle)
axd.loglog(r_hot_only/kpc, rho_wind_hot_only/(mu*mp),                              color = single_phase_color )
axP.loglog(r_hot_only/kpc, P_wind_hot_only/kb,                                     color = single_phase_color )
axZ.loglog(r_hot_only/kpc, np.ones_like(rho_wind_hot_only)*Z_wind_initial/Z_solar, color = single_phase_color )
axK.loglog(r_hot_only/kpc, K_wind_hot_only,                                        color = single_phase_color )

axMd.loglog(r_hot_only/kpc, Mdot_wind_hot_only, color = single_phase_color )
axEd.loglog(r_hot_only/kpc, Edot_wind_hot_only, color = single_phase_color )

cold_cloud_injection_radial_power = 6
cold_cloud_injection_radial_extent = 1.33*r0
cloud_radial_offest = 2e-2
irstart     = np.argmin(np.abs(r_hot_only-r0*(1.0+cloud_radial_offest)))
r_init      = r_hot_only[irstart]
v_init      = v_wind_hot_only[irstart]
rho_init    = rho_wind_hot_only[irstart]
P_init      = P_wind_hot_only[irstart]

for i_cloud,log_M_cloud0 in enumerate(log_cloud_masses):
    M_cloud0    = 10**np.array([log_M_cloud0])*Msun
    Z_cloud0    = np.ones_like(M_cloud0) * 1 * Z_solar #10**-1.0
    v_cloud0    = np.ones_like(M_cloud0) * 10**1.5 * km/s # * v_init * v_factor #
    Mdot_cold0  = eta_M_cold_tot * SFR
    Ndot_cloud0 = Mdot_cold0 / M_cloud0

    supersonic_initial_conditions = np.r_[v_init, rho_init, P_init, Z_wind_initial*rho_init, M_cloud0, v_cloud0, Z_cloud0]
    sol = solve_ivp(Wind_Evo, [r_init, 10**2*r0], supersonic_initial_conditions, events=[supersonic], dense_output=True, rtol=1e-12)
    print(sol.message)
    print(sol.t_events)

    r           = sol.t
    v_wind      = sol.y[0]
    rho_wind    = sol.y[1]
    P_wind      = sol.y[2]
    rhoZ_wind   = sol.y[3]
    M_cloud     = sol.y[4]
    v_cloud     = sol.y[5]
    Z_cloud     = sol.y[6]

    cloud_Mdots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud / (Msun/yr))
    Mdot_wind    = Omwind * r**2 * rho_wind * v_wind/(Msun/yr)
    cs_wind      = np.sqrt(gamma * P_wind / rho_wind)
    T_wind       = P_wind/kb/(rho_wind/(mu*mp))
    K_wind       = (P_wind/kb) / (rho_wind/(mu*mp))**gamma

    Pdot_wind    = Omwind * r**2 * rho_wind * v_wind**2/(1e5*Msun/yr)
    Pdot_wind_P  = Omwind * r**2 * (rho_wind * v_wind**2 + P_wind)/(1e5*Msun/yr)
    cloud_Pdots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * v_cloud / (1e5 * Msun/yr))

    Edot_wind    = Omwind * r**2 * rho_wind * v_wind * (0.5 * v_wind**2 + 1.5 * cs_wind**2)/(1e5**2*Msun/yr)
    cloud_Edots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * (0.5 * v_cloud**2 + 2.5 * kb * T_cloud/(mu*mp)) / (1e5**2 * Msun/yr))

    axv.loglog(r/kpc, v_wind/1e5,                                                        color = cloud_colors[i_cloud] )
    axv.loglog(r/kpc, cs_wind/1e5,                                                       color = cloud_colors[i_cloud] ,ls = cs_linestyle )
    axv.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, v_cloud)/1e5,        color = cloud_colors[i_cloud] ,ls = cloud_linestyle )
    axd.loglog(r/kpc, rho_wind/(mu*mp),                                                  color = cloud_colors[i_cloud] )
    axP.loglog(r/kpc, P_wind/kb,                                                         color = cloud_colors[i_cloud] )
    axZ.loglog(r/kpc, rhoZ_wind / rho_wind / Z_solar,                              color = cloud_colors[i_cloud] )
    axZ.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, Z_cloud)/Z_solar,    color = cloud_colors[i_cloud] ,ls = cloud_linestyle )
    axK.loglog(r/kpc, K_wind,                                                            color = cloud_colors[i_cloud] )
    axM.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)/Msun,       color = cloud_colors[i_cloud] )
    if sol.status == 1:
        axM.scatter(r[-1]/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)[-1]/Msun,  color = cloud_colors[i_cloud] , marker='x', s=8)
    axM.text(cold_cloud_injection_radial_extent*0.5/kpc, 0.55*np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)[0]/Msun, 
        r"$\xi_{\rm init} = "+str(np.round(cloud_ksi(r_init, supersonic_initial_conditions),2))+r"$", 
        ha='center', va='top', color = cloud_colors[i_cloud], fontsize=6)

    axX.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_ksi(r, sol.y)), color = cloud_colors[i_cloud] )


    axMd.loglog(r/kpc, Mdot_wind,                                                       color = cloud_colors[i_cloud] )
    axMd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Mdots[0]),  color = cloud_colors[i_cloud] , ls = cloud_linestyle)


    axEd.loglog(r/kpc, Edot_wind,                                                       color = cloud_colors[i_cloud] )
    axEd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Edots[0]),      color = cloud_colors[i_cloud] , ls = cloud_linestyle)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cs_linestyle),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axv.legend(custom_lines, [r'$v_r$', r'$c_s$', r'$v_{\rm cl}$'],loc='lower center',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axZ.legend(custom_lines, [r'$Z$', r'$Z_{\rm cl}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axMd.legend(custom_lines, [r'${\rm hot}$', r'${\rm cold}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

axv.set_ylabel(r'$v \; [{\rm km/s}]$')
axd.set_ylabel(r'$n \; [{\rm cm}^{-3}]$')
axP.set_ylabel(r'$P \; [{\rm K cm}^{-3}]$')
axZ.set_ylabel(r'$Z \; [Z_\odot]$')
axK.set_ylabel(r'$K \; [{\rm K cm}^{2}]$')
axM.set_ylabel(r'$M_{\rm cl} \; [M_\odot]$')
axX.set_ylabel(r'$\xi = r_{\rm cl} / v_{\rm turb} t_{\rm cool}$')
axX.axhline(1, color='grey', lw=0.5, ls=':')

axMd.set_ylabel(r'$\dot{M} \; [M_\odot/{\rm yr}]$')
axEd.set_ylabel(r'$\dot{E} \; [{\rm km}^2/{\rm s}^2 \; M_\odot/{\rm yr}]$')

axMd.set_xlabel(r'$r\; [{\rm kpc}]$')
axEd.set_xlabel(r'$r\; [{\rm kpc}]$')
axK.set_xlabel(r'$r\; [{\rm kpc}]$')

axM.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))

axd.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1])
axd.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
axd.set_ylim(bottom = 3e-6)
# axK.set_ylim((2e8,2e10))
axP.set_yticks([1e1,1e2,1e3,1e4,1e5,1e6,1e7])
axP.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
axP.set_ylim(bottom = 3e0)
axZ.set_yticks([1,2])
axZ.set_yticklabels([r"$1$",r"$2$"])
axZ.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# axK.set_ylim(top=1.1e10)
axK.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axM.set_ylim((8, 2e6))
axM.set_yticks([1e1,1e2,1e3,1e4,1e5,1e6])
axMd.set_ylim((0.5, 12))
axMd.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


axEd.set_ylim((2e5,2e7))
axEd.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

# axX.set_ylim((3e-2, 4))
axX.set_ylim(bottom=3e-2)
fig.set_size_inches(6.5,5)
plt.savefig('CaseI_Example_with_fluxes_ksi_SFR'+str(np.round(SFR/(Msun/yr),1))+'_etaE1_etaM'+str(np.round(eta_M,2))+'_etaMcold'+str(np.round(eta_M_cold_tot,2))+'_vcl3e1_Zcl1_distributed_injection_to_1+'+str(cloud_radial_offest)+'_'+str(np.round(cold_cloud_injection_radial_extent/r0,3))+'r0_power'+str(cold_cloud_injection_radial_power)+'_cloud_mass_variation_2_to_6_Mdot_coefficient033_vcirc'+str(np.round(v_circ0/1e5))+'_cooling_split_perfect.pdf',dpi=200,bbox_inches='tight')
# plt.savefig('CaseI_Example_with_fluxes_ksi_SFR'+str(np.round(SFR/(Msun/yr),1))+'_etaE1_etaM01_etaMcold'+str(np.round(eta_M_cold_tot,2))+'_vcl3e1_Zcl1_distributed_injection_to_1+1e-2_3r0_cloud_mass_variation_2_to_6_Mdot_coefficient033.png',dpi=400,bbox_inches='tight')
plt.clf()

plt.close('all')



































SFR = 20 * Msun/yr

eta_M              = 0.5
eta_M_cold_tot     = 0.5


### METALLICITY
Z_solar = 0.02

# feedback and SF props
E_SN  = 1e51
mstar = 100*Msun
M_cloud_min = 1e-2*Msun

## model choices
CoolingAreaChiPower         =  0.5 
ColdTurbulenceChiPower      = -0.5 
TurbulentVelocityChiPower   =  0.0 
geometric_factor            = 1.0
Mdot_coefficient            = 1.0/3.0
Cooling_Factor              = 1.0
drag_coeff                  = 0.5
f_turb0                     = 10**-1.0
v_circ0                     = 150e5 # gravitational potential assuming isothermal potential


eta_E              = 1
r0                 = 300*pc
Z_wind_initial     = 2.0 * Z_solar
half_opening_angle = np.pi/2
Omwind             = 4*np.pi*(1.0 - np.cos(half_opening_angle))

Mdot        = eta_M * SFR
Edot        = eta_E * (E_SN/mstar) * SFR

# properties at r0 if no clouds + gravity
epsilon     = 1e-5
Mach0       = 1.0 + epsilon
v0          = np.sqrt(Edot/Mdot)*(1/((gamma-1)*Mach0) + 1/2.)**(-1/2.)
rho0        = Mdot/(Omwind*r0**2 * v0)
P0          = rho0*v0**2 / Mach0**2 / gamma
rhoZ0       = rho0 * Z_wind_initial
print( "v_wind = %.1e km/s  n_wind = %.1e cm^-3  P_wind = %.1e kb K cm^-3" %(v0/1e5, rho0/(mu*mp), P0/kb))

Edot_per_Vol = Edot / (4/3. * np.pi * r0**3) # source terms from SN
Mdot_per_Vol = Mdot / (4/3. * np.pi * r0**3) # source terms from SN

r_init = 100*pc

dv_dr0, drho_dr0, dP_dr0 = Hot_Wind_Evo(r0, np.r_[v0, rho0, P0])

dlogvdlogr   = dv_dr0 * r0/v0
dlogrhodlogr = drho_dr0 * r0/rho0
dlogPdlogr   = dP_dr0 * r0/P0
dlogr0       = 1e-8

v0_sub   = 10**(np.log10(v0) - dlogvdlogr * dlogr0)
rho0_sub = 10**(np.log10(rho0) - dlogrhodlogr * dlogr0)
P0_sub   = 10**(np.log10(P0) - dlogPdlogr * dlogr0)

sol = solve_ivp(Hot_Wind_Evo, [10**(np.log10(r0)-dlogr0),r_init], np.r_[v0_sub, rho0_sub, P0_sub], 
    events=[supersonic,cold_wind], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_init    = sol.t[-1]
v_init    = sol.y[0][-1]
rho_init  = sol.y[1][-1]
P_init    = sol.y[2][-1]
rhoZ_init = rho_init * Z_wind_initial

v0_sup   = 10**(np.log10(v0) + dlogvdlogr * dlogr0)
rho0_sup = 10**(np.log10(rho0) + dlogrhodlogr * dlogr0)
P0_sup   = 10**(np.log10(P0) + dlogPdlogr * dlogr0)

sol_sup = solve_ivp(Hot_Wind_Evo, [10**(np.log10(r0)+dlogr0),10**2*r0], np.r_[v0_sup, rho0_sup, P0_sup], 
    events=[supersonic], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_hot_only         = np.append(sol.t[::-1], sol_sup.t)
v_wind_hot_only    = np.append(sol.y[0][::-1], sol_sup.y[0])
rho_wind_hot_only  = np.append(sol.y[1][::-1], sol_sup.y[1])
P_wind_hot_only    = np.append(sol.y[2][::-1], sol_sup.y[2])

Mdot_wind_hot_only  = Omwind*r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only/(Msun/yr)
cs_wind_hot_only    = np.sqrt(gamma * P_wind_hot_only / rho_wind_hot_only)
T_wind_hot_only     = P_wind_hot_only/kb / (rho_wind_hot_only/(mu*mp))
K_wind_hot_only  = (P_wind_hot_only/kb) / (rho_wind_hot_only/(mu*mp))**gamma
Pdot_wind_hot_only = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only**2/(1e5*Msun/yr)
Pdot_wind_hot_only_P = Omwind * r_hot_only**2 * (rho_wind_hot_only * v_wind_hot_only**2 + P_wind_hot_only)/(1e5*Msun/yr)
Edot_wind_hot_only = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only * (0.5 * v_wind_hot_only**2 + 1.5 * cs_wind_hot_only**2)/(1e5**2*Msun/yr)


# cold cloud initial properties
T_cloud             = 1e4
log_cloud_masses    = np.array([2,3,4,5,6])
N_cloud_masses      = len(log_cloud_masses)
cloud_colors        = cmr.take_cmap_colors('cmr.infinity_s', N_cloud_masses, cmap_range=(0.1, 0.9), return_fmt='hex')

fig,((axv,axZ,axd),(axM,axX,axP),(axMd,axEd,axK)) = plt.subplots(3,3,sharex=True,constrained_layout=True)
single_phase_color  = 'gray'
cs_linestyle        = ':'
cloud_linestyle     = '--'
axv.loglog(r_hot_only/kpc, v_wind_hot_only/1e5,                                    color = single_phase_color )
axv.loglog(r_hot_only/kpc, cs_wind_hot_only/1e5,                                   color = single_phase_color , ls = cs_linestyle)
axd.loglog(r_hot_only/kpc, rho_wind_hot_only/(mu*mp),                              color = single_phase_color )
axP.loglog(r_hot_only/kpc, P_wind_hot_only/kb,                                     color = single_phase_color )
axZ.loglog(r_hot_only/kpc, np.ones_like(rho_wind_hot_only)*Z_wind_initial/Z_solar, color = single_phase_color )
axK.loglog(r_hot_only/kpc, K_wind_hot_only,                                        color = single_phase_color )

axMd.loglog(r_hot_only/kpc, Mdot_wind_hot_only, color = single_phase_color )
axEd.loglog(r_hot_only/kpc, Edot_wind_hot_only, color = single_phase_color )

cold_cloud_injection_radial_power = 6
cold_cloud_injection_radial_extent = 1.33*r0
cloud_radial_offest = 2e-2
irstart     = np.argmin(np.abs(r_hot_only-r0*(1.0+cloud_radial_offest)))
r_init      = r_hot_only[irstart]
v_init      = v_wind_hot_only[irstart]
rho_init    = rho_wind_hot_only[irstart]
P_init      = P_wind_hot_only[irstart]

for i_cloud,log_M_cloud0 in enumerate(log_cloud_masses):
    M_cloud0    = 10**np.array([log_M_cloud0])*Msun
    Z_cloud0    = np.ones_like(M_cloud0) * 1 * Z_solar #10**-1.0
    v_cloud0    = np.ones_like(M_cloud0) * 10**1.5 * km/s # * v_init * v_factor #
    Mdot_cold0  = eta_M_cold_tot * SFR
    Ndot_cloud0 = Mdot_cold0 / M_cloud0

    supersonic_initial_conditions = np.r_[v_init, rho_init, P_init, Z_wind_initial*rho_init, M_cloud0, v_cloud0, Z_cloud0]
    sol = solve_ivp(Wind_Evo, [r_init, 10**2*r0], supersonic_initial_conditions, events=[supersonic,cold_wind], dense_output=True, rtol=1e-12)
    print(sol.message)
    print(sol.t_events)

    r           = sol.t
    v_wind      = sol.y[0]
    rho_wind    = sol.y[1]
    P_wind      = sol.y[2]
    rhoZ_wind   = sol.y[3]
    M_cloud     = sol.y[4]
    v_cloud     = sol.y[5]
    Z_cloud     = sol.y[6]

    cloud_Mdots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud / (Msun/yr))
    Mdot_wind    = Omwind * r**2 * rho_wind * v_wind/(Msun/yr)
    cs_wind      = np.sqrt(gamma * P_wind / rho_wind)
    T_wind       = P_wind/kb/(rho_wind/(mu*mp))
    K_wind       = (P_wind/kb) / (rho_wind/(mu*mp))**gamma

    Pdot_wind    = Omwind * r**2 * rho_wind * v_wind**2/(1e5*Msun/yr)
    Pdot_wind_P  = Omwind * r**2 * (rho_wind * v_wind**2 + P_wind)/(1e5*Msun/yr)
    cloud_Pdots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * v_cloud / (1e5 * Msun/yr))

    Edot_wind    = Omwind * r**2 * rho_wind * v_wind * (0.5 * v_wind**2 + 1.5 * cs_wind**2)/(1e5**2*Msun/yr)
    cloud_Edots  = (np.outer(Ndot_cloud0, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * (0.5 * v_cloud**2 + 2.5 * kb * T_cloud/(mu*mp)) / (1e5**2 * Msun/yr))

    axv.loglog(r/kpc, v_wind/1e5,                                                        color = cloud_colors[i_cloud] )
    axv.loglog(r/kpc, cs_wind/1e5,                                                       color = cloud_colors[i_cloud] ,ls = cs_linestyle )
    axv.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, v_cloud)/1e5,        color = cloud_colors[i_cloud] ,ls = cloud_linestyle )
    axd.loglog(r/kpc, rho_wind/(mu*mp),                                                  color = cloud_colors[i_cloud] )
    axP.loglog(r/kpc, P_wind/kb,                                                         color = cloud_colors[i_cloud] )
    axZ.loglog(r/kpc, rhoZ_wind / rho_wind / Z_solar,                              color = cloud_colors[i_cloud] )
    axZ.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, Z_cloud)/Z_solar,    color = cloud_colors[i_cloud] ,ls = cloud_linestyle )
    axK.loglog(r/kpc, K_wind,                                                            color = cloud_colors[i_cloud] )
    axM.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)/Msun,       color = cloud_colors[i_cloud] )
    if sol.status == 1:
        axM.scatter(r[-1]/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)[-1]/Msun,  color = cloud_colors[i_cloud] , marker='x', s=8)
    axM.text(cold_cloud_injection_radial_extent*0.5/kpc, 0.55*np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)[0]/Msun, 
        r"$\xi_{\rm init} = "+str(np.round(cloud_ksi(r_init, supersonic_initial_conditions),1))+r"$", 
        ha='center', va='top', color = cloud_colors[i_cloud], fontsize=6)

    axX.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_ksi(r, sol.y)), color = cloud_colors[i_cloud] )


    axMd.loglog(r/kpc, Mdot_wind,                                                       color = cloud_colors[i_cloud] )
    axMd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Mdots[0]),  color = cloud_colors[i_cloud] , ls = cloud_linestyle)


    axEd.loglog(r/kpc, Edot_wind,                                                       color = cloud_colors[i_cloud] )
    axEd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Edots[0]),      color = cloud_colors[i_cloud] , ls = cloud_linestyle)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cs_linestyle),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axv.legend(custom_lines, [r'$v_r$', r'$c_s$', r'$v_{\rm cl}$'],
    bbox_to_anchor=(0.4, 0), loc='lower center',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axZ.legend(custom_lines, [r'$Z$', r'$Z_{\rm cl}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axMd.legend(custom_lines, [r'${\rm hot}$', r'${\rm cold}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

axv.set_ylabel(r'$v \; [{\rm km/s}]$')
axd.set_ylabel(r'$n \; [{\rm cm}^{-3}]$')
axP.set_ylabel(r'$P \; [{\rm K cm}^{-3}]$')
axZ.set_ylabel(r'$Z \; [Z_\odot]$')
axK.set_ylabel(r'$K \; [{\rm K cm}^{2}]$')
axM.set_ylabel(r'$M_{\rm cl} \; [M_\odot]$')
axX.set_ylabel(r'$\xi = r_{\rm cl} / v_{\rm turb} t_{\rm cool}$')
axX.axhline(1, color='grey', lw=0.5, ls=':')

axMd.set_ylabel(r'$\dot{M} \; [M_\odot/{\rm yr}]$')
axEd.set_ylabel(r'$\dot{E} \; [{\rm km}^2/{\rm s}^2 \; M_\odot/{\rm yr}]$')

axMd.set_xlabel(r'$r\; [{\rm kpc}]$')
axEd.set_xlabel(r'$r\; [{\rm kpc}]$')
axK.set_xlabel(r'$r\; [{\rm kpc}]$')

axM.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))

axd.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1])
axd.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
axd.set_ylim(bottom = 3e-6)
# axK.set_ylim((2e8,2e10))
axP.set_yticks([1e1,1e2,1e3,1e4,1e5,1e6,1e7])
axP.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
axP.set_ylim(bottom = 3e0)
axZ.set_yticks([1,2])
axZ.set_yticklabels([r"$1$",r"$2$"])
axZ.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# axK.set_ylim(top=1.1e10)
axK.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axM.set_ylim((8, 3e6))
axM.set_yticks([1e1,1e2,1e3,1e4,1e5,1e6])
axMd.set_ylim(bottom=0.8)
axMd.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


axEd.set_ylim((2e5,2e7))
axEd.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

# axX.set_ylim((3e-2, 4))
axX.set_ylim(bottom=3e-2)
fig.set_size_inches(6.5,5)
plt.savefig('CaseI_Example_with_fluxes_ksi_SFR'+str(np.round(SFR/(Msun/yr),1))+'_etaE1_etaM'+str(np.round(eta_M,2))+'_etaMcold'+str(np.round(eta_M_cold_tot,2))+'_vcl3e1_Zcl1_distributed_injection_to_1+'+str(cloud_radial_offest)+'_'+str(np.round(cold_cloud_injection_radial_extent/r0,3))+'r0_power'+str(cold_cloud_injection_radial_power)+'_cloud_mass_variation_2_to_6_Mdot_coefficient033_vcirc'+str(np.round(v_circ0/1e5))+'_cooling_split_perfect.pdf',dpi=200,bbox_inches='tight')
# plt.savefig('CaseI_Example_with_fluxes_ksi_SFR'+str(np.round(SFR/(Msun/yr),1))+'_etaE1_etaM01_etaMcold'+str(np.round(eta_M_cold_tot,2))+'_vcl3e1_Zcl1_distributed_injection_to_1+1e-2_3r0_cloud_mass_variation_2_to_6_Mdot_coefficient033.png',dpi=400,bbox_inches='tight')
plt.clf()

plt.close('all')











###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>###########<><><>
