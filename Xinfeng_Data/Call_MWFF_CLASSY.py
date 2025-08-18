#This is the same as Call_MWFF_CLASSY but I put it in python not in jupyter lab since multiprocessing cannot run there.
#This is another try of running emcee in parallel for only the test galaxy


import csv 
import emcee
import numpy as np
from Multiphase_Wind_Fitting_function_for_Classy import *
import time
from multiprocessing import Pool
import numpy as np
import glob
from scipy import integrate, interpolate
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.interpolate import interp1d
import matplotlib.font_manager
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import os
import corner
import matplotlib.ticker as ticker
import pandas as pd
import re
from scipy.io import readsav

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
#matplotlib.rcParams['text.usetex'] = True
#plt.rc('text', usetex=True) This requires install of the package.

#Here are my functions that are not related to emcee
#Read a space separated file
#Ncol is the total # of cols in the file.
#Nread is the total # of cols that you want to return.
#Line starting with # will be skipped
#Read a space separated file
#Ncol is the total # of cols in the file.
#Nread is the total # of cols that you want to return.
#Line starting with # will be skipped
def ReadcolXX(MasterFile, Ncol, Nread, delimiter):

    # Open the text file for reading
    with open(MasterFile, 'r') as f:

        #1. Create a csv reader object with space as the delimiter
        # skipinitialspace is to skip consecutive spaces
        reader = csv.reader(f, delimiter=delimiter,skipinitialspace=True)

        #2. Define a dictionary to store the different columns
        arr = {}
        for i in range(1,Nread+1):
            arr['arr'+str(i)] = [] #i.e., it will contain keys as arr1, arr2, arr3, etc
        #print(arr)

        # Loop through the rows and extract the columns
        for row in reader:
            if len(row) == Ncol:
                #print(f"One line = {row}")

                if row[0][0] == '#':
                    a = 1#placeholder
                    #print("This line is skipped since no data!")
                else:
                    for i in range(1,Nread+1):
                        arr['arr'+str(i)].append(row[i-1])


    if Nread == 1:
        return arr['arr1']
    if Nread == 2:
        return arr['arr1'], arr['arr2']
    if Nread == 3:
        return arr['arr1'], arr['arr2'], arr['arr3']
    if Nread == 4:
        return arr['arr1'], arr['arr2'], arr['arr3'], arr['arr4']
    if Nread == 5:
        return arr['arr1'], arr['arr2'], arr['arr3'], arr['arr4'], arr['arr5']
    if Nread == 6:
        return arr['arr1'], arr['arr2'], arr['arr3'], arr['arr4'], arr['arr5'], arr['arr6']
    if Nread == 7:
        return arr['arr1'], arr['arr2'], arr['arr3'], arr['arr4'], arr['arr5'], arr['arr6'], arr['arr7']
    if Nread == 8:
        return arr['arr1'], arr['arr2'], arr['arr3'], arr['arr4'], arr['arr5'], arr['arr6'], arr['arr7'], arr['arr8']
    if Nread == 9:
        return arr['arr1'], arr['arr2'], arr['arr3'], arr['arr4'], arr['arr5'], arr['arr6'], arr['arr7'], arr['arr8'], arr['arr9']
    if Nread == 10:
        return arr['arr1'], arr['arr2'], arr['arr3'], arr['arr4'], arr['arr5'], arr['arr6'], arr['arr7'], arr['arr8'], arr['arr9'], arr['arr10']
    if Nread == 11:
        return arr['arr1'], arr['arr2'], arr['arr3'], arr['arr4'], arr['arr5'], arr['arr6'], arr['arr7'], arr['arr8'], arr['arr9'], arr['arr10'], arr['arr11']

#Extract string from the model name
def extract_method(WhichMod):
    match = re.search(r'00\d+_(\S+)', WhichMod)
    return match.group(1) if match else None

def calculate_parameter_errors(samples, blobs, ThetaMaxPath, galaxy_properties, ifDebug):
    """
    Compute and output the best-fit parameters and their uncertainties.
    """

    # 1. Fit parameters
    median = np.median(samples, axis=0)
    lower_ci = np.percentile(samples, 16, axis=0)
    upper_ci = np.percentile(samples, 84, axis=0)

    # 2. Extract blobs and clean NaNs
    v_clouds   = blobs["mean_v_cloud"] / 1e5  # cm/s → km/s
    v_widths   = blobs["v_width_cloud"] / 1e5
    nh_clouds  = blobs["N_cloud"]

    if ifDebug == 1:
        print(f"Before NaN removal: v_clouds = {v_clouds}")

    valid = (~np.isnan(v_clouds)) & (~np.isnan(v_widths)) & (~np.isnan(nh_clouds))
    v_clouds_clean  = v_clouds[valid]
    v_widths_clean  = v_widths[valid]
    nh_clouds_clean = nh_clouds[valid]

    if ifDebug == 1:
        print(f"After NaN removal: v_clouds = {v_clouds_clean}")

    # 3. Compute percentiles, while Handle empty arrays gracefully
    if len(v_clouds_clean) == 0:
        v_med = v_lo = v_hi = np.nan
        vwidth_med = vwidth_lo = vwidth_hi = np.nan
        nh_med = nh_lo = nh_hi = np.nan
        if ifDebug:
            print("[Warning] All blob values are NaN — skipping percentiles.")
    else:
        v_med, v_lo, v_hi         = np.percentile(v_clouds_clean, [50, 16, 84])
        vwidth_med, vwidth_lo, vwidth_hi = np.percentile(v_widths_clean, [50, 16, 84])
        nh_med, nh_lo, nh_hi      = np.percentile(nh_clouds_clean, [50, 16, 84])

    if ifDebug == 1:
        print(f"v_med, v_lo, v_hi = {v_med, v_lo, v_hi}")
    
    # 4. Combine
    median    = np.append(median, [v_med, vwidth_med, nh_med])
    lower_ci  = np.append(lower_ci, [v_lo, vwidth_lo, nh_lo])
    upper_ci  = np.append(upper_ci, [v_hi, vwidth_hi, nh_hi])

    # 5. Names
    parameter_names = [
        'eta_M_hot',
        'eta_M_cold',
        'log_M_cl',
        'mean_v_cloud',
        'v_width_cloud',
        'total_N_cloud'
    ]

    # 6. Write file
    with open(ThetaMaxPath, 'w') as f:
        f.write('Parameter\tMedian\tLower CI\tUpper CI\n')

        for i, name in enumerate(parameter_names):
            f.write(f'{name}\t{median[i]:.5g}\t{lower_ci[i]:.5g}\t{upper_ci[i]:.5g}\n')


#functions to draw the fitting results in different ways
def MakePlots(oneobjname, sampler, folder_path, x, y, yerr, initial, galaxy_properties, Mdot, Pdot, Edot,\
             MdotErrUp, MdotErrDo, PdotErrUp, PdotErrDo, EdotErrUp, EdotErrDo, ifDebug, Method, ifPlotCLASSYIII, ifShowFig):     
    

    #1. Load the MCMC results from backend.h5 - Note I no longer save the sampler.h5 since it is not necessary.
    bkfilename = folder_path + "backend.h5"
    reader = emcee.backends.HDFBackend(bkfilename)
    
    # Get flattened chains and log probabilities
    samples = reader.get_chain(flat=True)
    log_probs = reader.get_log_prob(flat=True)
    
    # Find the sample with the highest log probability
    theta_max = samples[np.argmax(log_probs)]
    blobs = reader.get_blobs(flat=True)
    
    print('Theta max from argmax: ',theta_max)
    if ifDebug ==1:
        print(f"samples.shape = {samples.shape}") #32768, 3
        print(f"galaxy_properties = {galaxy_properties}")
        print("blobs.ndim =", blobs.ndim)

    #3. Make all figures
    FigurePath = os.path.join(folder_path, oneobjname+'_fit.pdf')
    pdf_pages = PdfPages(FigurePath)


    #3.1 x-y
    fig1, ax = plt.subplots()
    best_fit_model0 = calculate_moments(theta_max[0],theta_max[1],theta_max[2], galaxy_properties)  

    extra_const  = best_fit_model0[4]    #This is the extra pars in Drummond's model - note I need to have this line before the next. 
    bin_centers  = best_fit_model0[8]    #These are used to make dN/dv figure
    dN_dv_binned = best_fit_model0[9]

    if Method == None: #default method to fit three moments (or outflow rates without the constant term) of the galaxy
        
        best_fit_model = best_fit_model0[0:3]
        if ifDebug ==1:   
            print(f"best_fit moments without the omega*mu*mp*r term = {best_fit_model}")
            print(f"extra_const = {extra_const}")

        label1 = 'Observed Moments'
        label2 = 'Modeled Moments'
    elif Method == 'FitNH':
        
        best_fit_model = [np.log10(best_fit_model0[5]), np.log10(best_fit_model0[6]), np.log10(best_fit_model0[7])] #convert the outputs (vcl, sigma_cl, and NH_cl) from calculate_moments to log scale. Also convert sigma to FWHM
        if ifDebug ==1:   
            print(f"best_fit V_cl, FWHM_cl, NH_cl in cgs unit and log scale = {best_fit_model}")
            print(f"extra_const = {extra_const}")
            
        label1 = 'Observed V, FWHM, and NH'
        label2 = 'Modeled V, FWHM, and NH'            
    else:
        raw()


    #ax.plot(x,y,label='Observed Moments', marker='o', linestyle='', markersize=10)
    ax.errorbar(x, y, yerr=yerr, label=label1, marker='o', linestyle='', markersize=8, alpha=0.5)
    
    ax.plot(x,best_fit_model,label=label2, marker='D', linestyle='', markersize=8)
    
    ax.set_yscale('log')
    ax.legend()
    
    ax.set_xlabel(r'$N_{th}$ Input Parameter')
    ax.set_ylabel('Values of the Input Parameter')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Set integer ticks on the x-axis
    #Latex example: r'$\alpha_{\mathrm{sub}}$ (m/s$^2$)'

    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    
    #Also add text for Mdot, Pdot, Edot
    if Method == None:
        ax.text(x[0]+0.1, y[0], r'$\dot{M}_{out} = $'+str(Mdot)+' M$_{\odot}/yr$', color='b', ha='left', va='center', fontsize=9)
        ax.text(x[1]+0.1, y[1], r'$Log (\dot{P}_{out}) = $'+str(Pdot)+" dynes", color='b', ha='left', va='center', fontsize=9)
        ax.text(x[2]-0.1, y[2], r'$Log (\dot{E}_{out}) = $'+str(Edot)+" ergs/s", color='b', ha='right', va='center', fontsize=9)

        # Calculate y positions based on percentage of the total range
        # log percentage
        y_40 = y_min * (y_max / y_min) ** 0.40
        y_50 = y_min * (y_max / y_min) ** 0.50
        y_60 = y_min * (y_max / y_min) ** 0.60

    elif Method == 'FitNH':
        ax.text(x[0]+0.1, y[0], r'$\dot{V}_{out} = $'+str(round(10**y[0]/1E5,2))+r' km $s^{-1}$', color='b', ha='left', va='center', fontsize=9)
        ax.text(x[1]+0.1, y[1], r'$\dot{FWHM}_{out} = $'+str(round(10**y[1]/1E5,2))+r" km $s^{-1}$", color='b', ha='left', va='center', fontsize=9)
        ax.text(x[2]-0.1, y[2], r'$Log (\dot{N}_{H,out}) = $'+str(y[2])+r" $cm^{-2}$", color='b', ha='right', va='center', fontsize=9)

        #linear percentage
        y_40 = y_min + 0.40 * (y_max - y_min)
        y_50 = y_min + 0.50 * (y_max - y_min)
        y_60 = y_min + 0.60 * (y_max - y_min)
    
        
    #Note galaxy_properties is different from outside since I dont times the unit
    # Add text at these positions
    ax.text(x[0], y_40, r'$SFR = $'+str(round(galaxy_properties[0]/(Msun/yr),2))+' M$_{\odot}/yr$', color='r', ha='left', va='center', fontsize=9)
    
    ax.text(x[0], y_50, r'$R_{*} = $'+str(round(galaxy_properties[1]/pc,2))+' pc', color='r', ha='left', va='center', fontsize=9)
    
    ax.text(x[0], y_60, r'$V_{cir} = $'+str(round(galaxy_properties[2]/(km/s),2))+' km/s',color='r', ha='left', va='center', fontsize=9)

    #TBD: add the errorbars into the output later.
    #MdotErrUp, MdotErrDo, PdotErrUp, PdotErrDo, EdotErrUp, EdotErrDo
    
    pdf_pages.savefig(fig1)

    #3.2 Corner plot
    #Note these parameters are all initial values since they are direct inputs to the MCMC
    # Need to be consistent with theta in log_likelihood()
    labels = [r'$\eta_{M,hot,0}$',r'$\eta_{M,cold,0}$',r'Log $M_{cl,0}$'] 
    
    # Compute the 95th percentile of each parameter to avoid overly large upper bounds
    param_percentiles = np.percentile(samples, 99, axis=0)  # Get 95th percentile for each parameter
    
    # Define a more reasonable range
    param_ranges = [(-0.1, 1.1*param_percentiles[0]),  # eta_M hot
                    (-0.1, 1.1*param_percentiles[1]),  # eta_M cold
                    (0.0, 0.1+param_percentiles[2])]   # log_M_cl 

    # Define colors for contours
    contour_colors = ["navy", "royalblue", "skyblue"] #blue-ish
    #contour_colors = ["darkred", "orangered", "gold"] #hot-cold

    fig2 = corner.corner(samples, 
                     show_titles=True, 
                     labels=labels, 
                     plot_datapoints=True, 
                     quantiles=[0.16, 0.5, 0.84], 
                     range=param_ranges,  # Apply the custom axis limits
                    contour_kwargs={"colors": contour_colors})  # Apply custom colors

    pdf_pages.savefig(fig2)

    #3.3 Calculate some params and output
    #3.3.1 Calculate median and confidence intervals (e.g., 16th and 84th percentiles)
    #This is to calculate central value of the posterior distribution - unlike what I get from theta_max above, which are the solutions at the highest posterior probability
    ThetaMaxPath = os.path.join(folder_path, oneobjname+'_fit.txt')
    calculate_parameter_errors(samples, blobs, ThetaMaxPath, galaxy_properties, ifDebug)

    #3.3.2 Add initial parameters to extra_const and write all parameters to a file
    #Note I only do this when you fit, since I may change the initial values later but dont run the fit.
    if ifMCMC == 1:   
        initial_par = ['eta_M', 'eta_M_cold_tot', 'log_M_cloud0']
        merged_params = extra_const.copy()  
        for key, value in zip(initial_par, initial):
            merged_params[key] = value

        par_path = os.path.join(folder_path, oneobjname+'_par.txt')
        
        with open(par_path, "w") as file:
            for key, value in merged_params.items():
                file.write(f"{key}: {value}\n")

    #3.4 dN/dV plot

    #3.4.1 dN/dV model Convert bin_centers to km/s and take negative for blueshift (if desired)
    x_profile = -1 * np.array(bin_centers) / 1e5 #Convert to km/s and blueshifted velocity
    y_profile = dN_dv_binned*1e5   # already in cgs units, then I convert from cm^-2/(cm/s) to cm^-2/(km/s)
    
    fig3, ax3 = plt.subplots()
    fonts = 12
    
    ax3.step(x_profile, y_profile, where='mid', color='red',label='Best Fit Model')


    #3.4.1.2.1 Print the best fit parameters in the figure.
    
    x_min, x_max = np.nanmin(x_profile), np.nanmax(x_profile)
    y_min, y_max = np.nanmin(y_profile), np.nanmax(y_profile)

    #3.4.1.2.1 These are the solutions at max(logP) - not necessarily match the corner plots?
    #ax3.text(x_min + (x_max - x_min)*0.15, y_min + (y_max - y_min)*1.2,  "Best-fit Params:", color='k', ha='left', va='center', fontsize=12) 
    #ax3.text(x_min + (x_max - x_min)*0.15, y_min + (y_max - y_min)*1.13,  labels[0]+" = "+str(round(theta_max[0],2)), color='k', ha='left', va='center', fontsize=12)
    #ax3.text(x_min + (x_max - x_min)*0.15, y_min + (y_max - y_min)*1.06, labels[1]+" = "+str(round(theta_max[1],2)), color='k', ha='left', va='center', fontsize=12)
    #ax3.text(x_min + (x_max - x_min)*0.15, y_min + (y_max - y_min)*0.99, labels[2]+" = "+str(round(theta_max[2],2)),color='k', ha='left', va='center', fontsize=12)

    #3.4.1.2.2 These are the contour's value and errors:
    medians = np.median(samples, axis=0)
    lower_errors = medians - np.percentile(samples, 16, axis=0) # Calculate 1sigma errors
    upper_errors = np.percentile(samples, 84, axis=0) - medians
    
    param_texts = [
        rf"$\eta_{{M,\mathrm{{hot,0}}}} = {medians[0]:.2f}^{{+{upper_errors[0]:.2f}}}_{{-{lower_errors[0]:.2f}}}$",
        rf"$\eta_{{M,\mathrm{{cold,0}}}} = {medians[1]:.2f}^{{+{upper_errors[1]:.2f}}}_{{-{lower_errors[1]:.2f}}}$",
        rf"$\log M_{{\mathrm{{cl,0}}}} = {medians[2]:.2f}^{{+{upper_errors[2]:.2f}}}_{{-{lower_errors[2]:.2f}}}$"
    ]
    
    ax3.text(x_min + (x_max - x_min)*0.00, y_min + (y_max - y_min)*1.7, "Best-fit Params:", color='k', ha='left', va='center', fontsize=fonts)
    for i, text in enumerate(param_texts):
        y_offset = 1.6 - i*0.12
        ax3.text(x_min + (x_max - x_min)*0.15, y_min + (y_max - y_min)*y_offset, text, color='k', ha='left', va='center', fontsize=fonts)
        

    

    # Optionally, set y-scale to log:
    # ax3.set_yscale('log')

    #3.4.2 dN/dV observation from each CLASSY galaxy
    ObsModelPlotted = 0 #initial, the observed model has not been plotted
    
    if ifPlotCLASSYIII == 1:
        
        H15Obj = ['J0055-0021', 'J0150+1308', 'J1113+2930', 'J1414+0540', 'J2103-0728']
        current_obj = oneobjname
        
        #3.4.2.1 Check if the object belongs to H15Obj
        if current_obj in H15Obj:
            OutputFolder = os.path.join('./Results/CLASSYIII/MeasureNH', 'Heckman15_' + current_obj)
            OutputFolder2 = os.path.join('Results', 'Heckman15_' + current_obj)
        else:
            OutputFolder = os.path.join('./Results/CLASSYIII/MeasureNH', current_obj)
            OutputFolder2 = os.path.join('Results', current_obj)

        #3.4.2.2. Check if the saved solution file exists
        sav_filepath = os.path.join('/Applications/Work-Desktop/CLASSY/Analysis/', 'SolvePI', OutputFolder2, 'PISolution.sav')
        
        
        if os.path.exists(sav_filepath):

            print(f"-----Plotting the observed dN/dV for object = {current_obj} -----")

            
            # Define filename for the plot output (without extension)
            filename = os.path.join(OutputFolder, current_obj + '_velocity_NH')
            
            
            #3.4.2.3 Read the PI solution IDL file from CLASSY III 
            # The .sav file is assumed to contain the variables: velocity, dv, and PISolutionCube.
            data = readsav(sav_filepath)
            velocity = data['velocity']       # 1D numpy array
            dv = data['dv']                   # not used further in this code snippet
            PISolutionCube = data['PISolutionCube']  # 3D numpy array with dimensions (nVel, 10, 3)
            PISolutionCube = np.transpose(PISolutionCube, (2, 1, 0)) #IDL and python arrays are reversed.
    
            
            #3.4.2.4. Extract arrays for logUh and logNh and their errors from PISolutionCube.
            logUh      = PISolutionCube[:, 0, 0]
            logUhErrUp = PISolutionCube[:, 0, 1]
            logUhErrDo = PISolutionCube[:, 0, 2]
            logNh      = PISolutionCube[:, 1, 0]
            logNhErrUp = PISolutionCube[:, 1, 1]
            logNhErrDo = PISolutionCube[:, 1, 2]

            if ifDebug == 1:
                print("velocity from CLOUDY models =", velocity)
                print(f"logUh = {logUh}")
                print(f"logUhErrUp = {logUhErrUp}")
                print(f"logNh = {logNh}")
                print(f"logNhErrUp = {logNhErrUp}")
                                
            if not (np.all(logUh == -100) and np.all(logNh == -100)): #This one has not (no PI solution)
                #print('test1')
                     
                #3.4.2.5. Set the x-range and determine indices within the desired velocity range.
                xRan = [np.min(velocity)-100, np.max(velocity)+100]
                if xRan[0] > -700:
                    xRan[0] == -700
                if xRan[1] < 200:
                    xRan[1] == 200
                    
                indexVel = np.where((velocity > xRan[0]) & (velocity < xRan[1]))[0]

                
                #3.4.2.6. Interpolate the various arrays to the subset velocity array for NH
                velNh = velocity[indexVel]
                logUH_inter       = np.interp(velNh, velocity, logUh)
                logUHErrUp_inter  = np.interp(velNh, velocity, logUhErrUp)
                logUHErrDo_inter  = np.interp(velNh, velocity, logUhErrDo)
                logNH_inter       = np.interp(velNh, velocity, logNh)
                logNHErrUp_inter  = np.interp(velNh, velocity, logNhErrUp)
                logNHErrDo_inter  = np.interp(velNh, velocity, logNhErrDo)
        
                #3.4.2.7 Exclude bins where NH or UH = -100
                valid_bins = (logNH_inter != -100) & (logUH_inter != -100)
                
                # Apply mask to velocity, NH, and UH arrays
                velNh = velNh[valid_bins]
                logNH_inter = logNH_inter[valid_bins]
                logNHErrUp_inter = logNHErrUp_inter[valid_bins]
                logNHErrDo_inter = logNHErrDo_inter[valid_bins]
                logUH_inter = logUH_inter[valid_bins]
                logUHErrUp_inter = logUHErrUp_inter[valid_bins]
                logUHErrDo_inter = logUHErrDo_inter[valid_bins]
                
                if len(velNh) != 0: #This one has valid PI solution bins

                    #print('test2')
                    #3.4.2.8 Here I calculate the dN/dv profile
                    dN_dv_binned = (10**logNH_inter[:-1]) / np.diff(velNh)
                    bin_centers = velNh #0.5 * (velNh[:-1] + velNh[1:])
        
                    yRan = [np.min(dN_dv_binned)*0.7, np.max(dN_dv_binned)*1.4]
        
                    #ax3.plot(bin_centers[:-1], dN_dv_binned, linestyle='--', color='black', linewidth=3, \
                    #         marker='o', markersize=5,label='Xu+22')

                    ax3.step(bin_centers[:-1], dN_dv_binned, where='mid', label='Observations')

                    # 3.4.2.9 Propagate errors for dN/dV
                    NH_upper = 10**(logNH_inter[:-1] + logNHErrUp_inter[:-1]/2)
                    NH_lower = 10**(logNH_inter[:-1] - logNHErrDo_inter[:-1]/2)
                    
                    dN_dv_binned_ErrUp = NH_upper / np.diff(velNh)
                    dN_dv_binned_ErrDo = NH_lower / np.diff(velNh)
                    
                    # Fill error region
                    ax3.fill_between(bin_centers[:-1],
                                     dN_dv_binned_ErrDo,
                                     dN_dv_binned_ErrUp,
                                     step='mid',
                                     color='lightblue',
                                     alpha=0.4,
                                     label='Obs Errors')

                    ObsModelPlotted = 1 #Record if the observed model has been plotted

                    # Plot vertical lines for NH integration boundary
                    plt.axvline(float(galaxy_properties[5]), color='gray', linestyle='--')
                    plt.axvline(float(galaxy_properties[6]), color='gray', linestyle='--')


    
    #3.4 Set plotting range
    if ObsModelPlotted == 1:

        combined_x_min = min(np.min(xRan), np.min(x_profile))
        combined_x_max = max(np.max(xRan), np.max(x_profile))

        combined_y_min = min(np.min(yRan), np.min(y_profile))
        combined_y_max = max(np.max(yRan), np.max(y_profile))

    else:
        combined_x_min = np.min(x_profile)
        combined_x_max = np.max(x_profile)

        combined_y_min = np.min(y_profile)
        combined_y_max = np.max(y_profile)

    #cap the maximum dN/dV
    if combined_y_max >1E19:
        combined_y_max = 1E19
        
    ax3.set_xlim(combined_x_min, combined_x_max)
    ax3.set_ylim(combined_y_min, combined_y_max)


    # 3.5 Plot the top x-axis as the radius from the model.
    #r_array0 is the full radius arry in cm. Unlike the bin_centers and dN_dv_binned, I need to filter the small r part as I did in calculate_moments()
    try:
        r_array0 = best_fit_model0[3].t
        ir10 = np.argmin(np.abs(r_array0 - 5*kpc))
        r_array       = r_array0[2:ir10]
    
        
        # 3.5.1. Interpolate velocity as a function of radius
        interp_func = interp1d(r_array/kpc, x_profile, bounds_error=False, fill_value="extrapolate")
    
        # 3.5.2. Create twin axis
        ax_top = ax3.twiny()
    
        # 3.5.3. Set top axis limits to match bottom axis
        ax_top.set_xlim(ax3.get_xlim())
    
        # 3.5.4. Set top axis label and tick params
        ax_top.set_xlabel('r (kpc)', fontsize=fonts)
        ax_top.tick_params(axis='both', which='major', labelsize=fonts)
    
        # 3.5.5. Add manual ticks at specific radius values
        r_plot = np.array([1.0, 0.5])  # radii in kpc
    
        # 3.5.5.1 Find corresponding x positions (velocity) for these radii
        x_ticks_manual = interp_func(r_plot)
    
        # 3.5.5.2 Set the ticks and labels
        ax_top.set_xticks(x_ticks_manual)
        ax_top.set_xticklabels([f"{r:.1f}" for r in r_plot])
    
        if ifDebug == 2:
            print(f"x_profile = {x_profile}")
            print(f"r_array = {r_array/kpc}")
    
            print(f"r_plot = {r_plot}")
            print(f"x_ticks_manual = {x_ticks_manual}")
            
    except:
        #This means the fitting failed in some ways
        #best_fit_model0 is the original outputs from calculate_moments, best_fit_model is the revised one with correct units.
        print(f"Warning: For object {oneobjname}, best_fit_model0[3] is not a solve_ivp object.")
        print(f"best_fit_model0 = {best_fit_model0}")
        print(f"Skipping plot of r_plot")
        
    


    
    #Save the figure for dN/dV into pages
    ax3.set_xlabel('Velocity (km/s)', fontsize=fonts)
    ax3.set_ylabel(r'dN/dV (cm$^{-2}$ / (km/s))', fontsize=fonts)
    #ax3.set_title('dN/dV Profile')
    ax3.legend(fontsize=fonts)
    ax3.tick_params(axis='both', which='major', labelsize=fonts)


    pdf_pages.savefig(fig3)

    #3.N Write the PDF document to the disk
    pdf_pages.close()  # Close the PDF properly
    if ifShowFig == 1: # Avoid showing figures fro all objects
        plt.show()
    
    plt.close('all')  # Close all open figures - you have to close figures so they can be shown in the screen.
                      # Moreover, opening too many windows will consume the buffers and cause errors.

# Read a range of columns using Excel-style column letters
def ReadExcel(FilePath):
    df = pd.read_excel(FilePath, usecols='A:O')


    objname2 = df['Objname']
    Mdot = df['Mdot_outflow (Msun/yr)']
    Pdot = df['log(Pdot_outflow) (dynes)']
    Edot = df['log(Edot_outflow) (erg/s)']

    Vout    = df['Vout (km/s)']
    FWHMout = df['FWHMout (km/s)']
    NHout   = df['log N_H out (cm^-2)']

    return objname2,Mdot,Pdot,Edot,Vout, FWHMout, NHout

# Read a range of columns using Excel-style column letters
def ReadExcel2(FilePath):
    df = pd.read_excel(FilePath, usecols='A:P')


    objname4 = df['objname']
    SFR4 = df['LogSFR']
  
    return objname4,SFR4


#Tell if a string is a number.
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#Split string like 0.5+0.2-0.1 to be value+-errors
#If the format does not match, it will return None, None, None.
def parse_errors(error_string):
    # Use a regular expression to find the value and errors, allowing for leading whitespace
    pattern = re.compile(r"\s*([+-]?[0-9]*\.?[0-9]+)([+-][0-9]*\.?[0-9]+)(-([0-9]*\.?[0-9]+))")
    

    match = pattern.match(error_string)
    if match:
        value = float(match.group(1))
        upper_error = float(match.group(2))
        lower_error = float(match.group(4))
        return value, upper_error, lower_error
    else:
        return None, None, None

def safe_divide(x, divisor):
    return x / divisor if x is not None else None
    
def clean_strings(string_list):
    cleaned_list = []
    for string in string_list:
        # Split the string by '^' sign and take the first part
        clean_string = string.split('^')[0]
        # Remove leading/trailing spaces and special characters
        clean_string = clean_string.strip().replace('--', '-').replace('+ ', '+').replace('- ', '-')
        cleaned_list.append(clean_string)
    return cleaned_list

    
#Read the measurements from CLASSY III paper and construct the observed moments array
def ReadObservations(MasterPATH):
    
    #MasterFile = os.path.join(MasterPATH, 'In1.1_CLASSYProcess2.txt') 
    #ID, objname, zobj = ReadcolXX(MasterFile, 20, 3,delimiter = '|')
    #print("objname = ",objname,len(objname))
    #print("zobj = ",zobj,len(zobj))

    #1.1 This file contains the outflow rates
    FilePath2 = os.path.join(MasterPATH, 'Out5.1_HLSP.xlsx')
    objname2, Mdot, Pdot, Edot, Vout, FWHMout, NHout = ReadExcel(FilePath2)
    

    #1.2 This file contains the R_50
    FilePath3 = os.path.join(MasterPATH, 'Out5.3_AncillaryParams.txt')    
    objname3, zobj3, r_50_arcsec, r_50_kpc, vcir_arr = ReadcolXX(FilePath3, 7,5,delimiter = '&')
    

    #1.3 This file contains SFR
    FilePath4 = os.path.join(MasterPATH, 'Out6.5_HLSP.xlsx')    
    objname4, SFR4 = ReadExcel2(FilePath4)

    #1.4 This file contains the measured v_peak and HWHM of the dN/dV profile
    FilePath5 = os.path.join('./Results/CLASSYIII/MeasureNH', 'Out10.0_NH_Profile_Info.txt')    
    objname5, mean_v_cloud, HWHM_cloud, NH_cloud, NH_int_left, NH_int_rigt = ReadcolXX(FilePath5, 6, 6,delimiter = ',')

    NMeaMethod = 2 # =1 will use the CLASSY III paper's Vout, FWHMout, and total NH
                   # =2 will use dN/dV profile's abs(Vout) and FWHMout, and the same NH as above (actually very similar as the ones measured from dN/dV)
                        #These values are measured in ./Calculate_NH_CLASSY_obs

    
    #print(f"objname5 = {objname5}")
    #print(f"mean_v_cloud = {mean_v_cloud}")
    #print(f"HWHM_cloud = {HWHM_cloud}")
    #print(f"NH_cloud = {NH_cloud}")


    
    #sanity check to ensure that the objectname are the same.
    objname2 = clean_strings(objname2) #remove extra - and ^{} for shorter names.
    objname3 = clean_strings(objname3)
    objname4 = clean_strings(objname4)
    objname5 = clean_strings(objname5)
    
    if np.array_equal(objname2, objname3) != True:
        STOP
    if np.array_equal(objname2, objname4) != True:
        STOP
    if np.array_equal(objname2, objname5) != True:
        STOP
        
    #print(f"objname3 = {objname3}")
    #print(f"objname5 = {objname5}")
    #raw
    
    #3. Define some params in my paper
    omega   = 1.0    #outflow solid angle. 
                                   #TBD: thinking about if I should use Omwind/(4*np.pi)*1
                                   #In CLASSY III, I used solid angle = 4pi, so the scaling is 1 here.
                                   #But for other Omwind != 4pi, you need to scale differently. e.g., 2 pi -> 0.5 as scaling.
    pi      = 3.1415
    miu     = 1.4        # average photon mass fraction
    cmTopc  = 1.0/pc     # 1cm to parsec
    kgTosun = 1.586e-23  # 1kg/s = A Msun/year
    gTosun  = kgTosun*1E-3
    mp      = 1.67373522381e-24

    #4. Calculate the moments
    #Outflows rates - used in model001 - 003
    #note I still output the actual Mdot, pdot, Edot to calculate the initial eta values 
    m1      = [0.0] * len(objname3)
    m2      = [0.0] * len(objname3)
    m3      = [0.0] * len(objname3)
    m1_err  = [0.0] * len(objname3)
    m2_err  = [0.0] * len(objname3)
    m3_err  = [0.0] * len(objname3)

    #v, sigma_v and N - used in model001_FitN
    #I used NX just to match the format of m1 - m3.
    N1      = [0.0] * len(objname3) #Vout or abs(mean_v_cloud) directly from dN/dv - FB model returns all positive v, which does not consider doppler effect
    N2      = [0.0] * len(objname3) #FWHMout or HWHM_cloud directly from dN/dv
    N3      = [0.0] * len(objname3) #Total NHout
    N1_err  = [0.0] * len(objname3) 
    N2_err  = [0.0] * len(objname3)
    N3_err  = [0.0] * len(objname3)

    Int_left= [0.0] * len(objname3) #Integration range left side for NH_tot (or NH_out here). Calculate from  ./Calculate_NH_CLASSY_obs
    Int_rigt= [0.0] * len(objname3)

    #galaxy properties
    vcir    = [0.0] * len(objname3)
    sfr     = [0.0] * len(objname3)   
    r50     = [0.0] * len(objname3) 

    #Errors - note used
    MdotErrUp     = [0.0] * len(objname3) 
    MdotErrDo     = [0.0] * len(objname3) 
    PdotErrUp     = [0.0] * len(objname3) 
    PdotErrDo     = [0.0] * len(objname3) 
    EdotErrUp     = [0.0] * len(objname3) 
    EdotErrDo     = [0.0] * len(objname3) 


    for obji in range(0,len(objname3)):
       
        

        #4.1 First try to split Mdot, Pdot, Edot into value and errors
        #If no valid outflow rates, it will return None.
        value1, upper_error1, lower_error1 = parse_errors(Mdot[obji])
        value2, upper_error2, lower_error2 = parse_errors(Pdot[obji])
        value3, upper_error3, lower_error3 = parse_errors(Edot[obji])
        value4, upper_error4, lower_error4 = parse_errors(vcir_arr[obji])
        value5, upper_error5, lower_error5 = parse_errors(Vout[obji])
        value6, upper_error6, lower_error6 = parse_errors(FWHMout[obji])
        value7, upper_error7, lower_error7 = parse_errors(NHout[obji])

        
        #4.2 Get the value and errors
        oneMdot    = [value1, upper_error1, lower_error1]        #Msun/yr
        onePdot    = [value2, upper_error2, lower_error2]        #dynes in log.
        oneEdot    = [value3, upper_error3, lower_error3]        #ergs/s in log. 10**(Edot[obji]) 
        onevcir    = [value4, upper_error4, lower_error4]        #km/s

        if NMeaMethod ==1:
            oneVout    = [value5, upper_error5, lower_error5]        #km/s
            oneFWHMout = [value6, upper_error6, lower_error6]        #km/s
            oneNHout   = [value7, upper_error7, lower_error7]        #cm^-2
        elif NMeaMethod == 2:
            #ABS is needed since 
            #  1) FB model does not consider doppler effect so all velocity >0
            #  2) I have -100 for galaxies iwth no PI solution, and they will be skipped since Mdot is None later.

            #06/06/2025: I change this to use CLASSYIII's errorbar
            #oneVout    = [abs(float(mean_v_cloud[obji])), upper_error5, lower_error5]      #peak velocity measured from dN/dV profile. 
                                                                                          
            #oneFWHMout = [abs(float(HWHM_cloud[obji])), safe_divide(upper_error6, 2.0), safe_divide(lower_error6, 2.0)]        #HWHM measured from dN/dV profile -> errorbar is half of FWHMerr from CLASSYIII
                                                                                                                               #safe_divide is to avoid None/2 in this case.
            #oneNHout   = [abs(float(NH_cloud[obji])), upper_error7, lower_error7]        #cm^-2

             #Binning of dN/dV is 40km/s, so error is fixed at 20km/s here
            oneVout    = [abs(float(mean_v_cloud[obji])), 20, 20]      #peak velocity measured from dN/dV profile. 
                                                                                          
            oneFWHMout = [abs(float(HWHM_cloud[obji])), 20, 20]        
                                                                                                                               
            oneNHout   = [abs(float(NH_cloud[obji])), upper_error7, lower_error7]        #cm^-2


            
            
            
            one_Int_left= NH_int_left[obji] #fill the integration ranges for NHout
            one_Int_rigt= NH_int_rigt[obji]            
        else:
            print(f"ERROR: you have not defined how to get the measurements of N1-N3! NMeaMethod = {NMeaMethod} not defined!")
            raw

        #print(f"oneVout = {oneVout}")
        #raw
        
        oner_50    = float(r_50_kpc[obji])                       #kpc
        r_50_cm    = 2.0*oner_50*1000.0/cmTopc

        #print(f"oneVout = {oneVout}")
        #print(f"oneFWHMout = {oneFWHMout}")
        #print(f"oneNHout = {oneNHout}")
        #raw
        
        #4.3 Calculate the moments
        #Note the input Mdot, Pdot, Edot are in different units
        #But the output m1, m2, m3 are all in cgs units.
        if oneMdot[0] != None:
            const      = omega*miu*mp*r_50_cm     #constant outside the integration: g*cm
            
            m1[obji]   = oneMdot[0]/gTosun  / const  #s^-1 cm^-1 = dN_H/dv *v
            m2[obji]   = 10**onePdot[0]/const        #= dN_H/dv *v^2
            m3[obji]   = 10**oneEdot[0]/0.5/const    #= dN_H/dv *v^3
            
            m1_err[obji]   = (oneMdot[1] + oneMdot[2])/2.0/gTosun  / const                           #(upper error +lower error)/2
            #m2_err[obji]   = (10**(onePdot[0] + onePdot[1]) - 10**(onePdot[0]-onePdot[2]))/2.0/const #(upper value -lower value) /2, note upper value = value +-error
            #m3_err[obji]   = (10**(oneEdot[0] + oneEdot[1]) - 10**(oneEdot[0]-oneEdot[2]))/2.0/0.5/const 
            m2_err[obji]   =m1_err[obji]/m1[obji] * m2[obji]  #the above errors propagration fails for large errorbars in log
            m3_err[obji]   =m1_err[obji]/m1[obji] * m3[obji]  #so I just assum Pdot and Edot has similar error percentage as Mdot.

            #The results from MWFF are in cgs unit and linear: v, sigma, NH = (20235283.643189132, 21940751.96652906, 4.2525979069319023e+21)
            N1[obji]   = np.log10(oneVout[0]*10**5)    #logVout in cm/s
            N2[obji]   = np.log10(oneFWHMout[0]*10**5) #logFWHMout in cm/s
            N3[obji]   = oneNHout[0]                   #logNH - already in log scale and in cm^-2
            Int_left[obji] = one_Int_left
            Int_rigt[obji] = one_Int_rigt

            #N1_err does not need to be converted to cgs since now it is in log scale.
            N1_err[obji] = (oneVout[1] + oneVout[2]) / (2.0 * oneVout[0] * np.log(10))   #error_log = error_linear/value_linear/ln(10)
            N2_err[obji] = (oneFWHMout[1] + oneFWHMout[2]) / (2.0 * oneFWHMout[0] * np.log(10))  
            N3_err[obji] = (oneNHout[1] + oneNHout[2])/2.0

            #raw#TBD change the unit and likely to log scale here
        
        else:
            m1[obji]       = None
            m2[obji]       = None
            m3[obji]       = None          
            m1_err[obji]   = None
            m2_err[obji]   = None
            m3_err[obji]   = None   

            N1[obji]   = None   
            N2[obji]   = None   
            N3[obji]   = None   
            Int_left[obji] = None
            Int_rigt[obji] = None
            
            N1_err[obji]   = None
            N2_err[obji]   = None   
            N3_err[obji]   = None   

        
        #4.4 Store other galaxy properties: I dont need errorbars here
        vcir[obji] = onevcir[0]
        sfr[obji]  = 10**SFR4[obji]
        r50[obji]  = oner_50

        #4.5 Store error bars for MPE dot. Note I overwrite the arrays for values.
        Mdot[obji]          = value1
        MdotErrUp[obji]     = upper_error1
        MdotErrDo[obji]     = lower_error1

        Pdot[obji]          = value2
        PdotErrUp[obji]     = upper_error2
        PdotErrDo[obji]     = lower_error2

        Edot[obji]          = value3
        EdotErrUp[obji]     = upper_error3
        EdotErrDo[obji]     = lower_error3

    
        #sanity checks
        ifCheck = 0
        if ifCheck == 1:     
            objicheck = 6
            if obji == objicheck:
                print(f"objname3 = {objname3[obji]}")
                print(f"Mdot, Pdot, Edot, r_50 = {oneMdot[0], onePdot[0], oneEdot[0],oner_50}")
                print(f"Mdot, Pdot, Edot, r_50 errup = {oneMdot[1], onePdot[1], oneEdot[1],oner_50}")
                print(f"Mdot, Pdot, Edot, r_50 errdo = {oneMdot[2], onePdot[2], oneEdot[2],oner_50}")
                print(f"m1, m2, m3, const = {m1[obji],m2[obji],m3[obji],const}")
                print(f"m1, m2, m3 err = {m1_err[obji],m2_err[obji],m3_err[obji]}")
                print(f"SFR, Vcir = {sfr[obji], vcir[obji]}")
    
                print(f"N1, N2, N3 value = {N1[obji],N2[obji],N3[obji]}")
                print(f"N1, N2, N3 err = {N1_err[obji],N2_err[obji],N3_err[obji]}")
                print(f"NH integration range = {Int_left[obji], Int_rigt[obji]}")
                raw


    
    return objname3, zobj3, m1, m1_err, m2, m2_err, m3, m3_err, N1, N1_err, N2, N2_err, N3, N3_err, Int_left, Int_rigt,r50, sfr, vcir, Mdot, Pdot, Edot, MdotErrUp, MdotErrDo, PdotErrUp, PdotErrDo, EdotErrUp, EdotErrDo



#2. Here are emcee fitting functions
#2.1 This one calls Drummond's model
#theta are the fitted variables, while galaxy_properties are fixed input params.
#Added the return of blob to store the MCMC chain for them, see https://emcee.readthedocs.io/en/stable/user/blobs/
def log_likelihood(theta, x, y, yerr, galaxy_properties, Method):

    small_num = -np.inf #-1e10 instead of -np.inf cannot fit well
    
    eta_M, eta_M_cold_tot, log_M_cloud0 = theta
    first_moment, second_moment, third_moment, sol, extra_const,mean_v_cloud,v_width_cloud,N_cloud, _, _ = calculate_moments(eta_M, eta_M_cold_tot, log_M_cloud0, galaxy_properties)

    if Method == None:  #default method to fit three moments (or outflow rates without the constant term) of the galaxy
        ymodel0 = [first_moment, second_moment, third_moment]
        ymodel  = [first_moment, second_moment, third_moment]
        
    elif Method == 'FitNH':
        ymodel0 = [mean_v_cloud, v_width_cloud, N_cloud]

        if all(val > 0 for val in ymodel0):
            ymodel = [np.log10(mean_v_cloud), np.log10(v_width_cloud), np.log10(N_cloud)]
        else: 
            return small_num, (first_moment, second_moment, third_moment, mean_v_cloud, v_width_cloud, N_cloud)

            
    #Notes:
    #1. Both y and ymodel should only have three parameter: m1, m2, m3
    #2. If the modeled moments = -1, -3, -4, -5, it means the code breaks, so I need to return -np.inf
    #3. Calculate_moments now returns cgs unit.

    
    if np.any(np.isnan(ymodel0)):
        print(f"Warning: \
                    ymodel = {ymodel} contains NaN! \
                    eta_M, eta_M_cold_tot, log_M_cloud0 = {eta_M, eta_M_cold_tot, log_M_cloud0}, \
                    galaxy_properties = {galaxy_properties}")
        STOP_XX()

    else:
        try:
            lnlike = -0.5 * np.sum(((np.array(y) - np.array(ymodel)) / np.array(yerr)) ** 2)
            return lnlike, (first_moment, second_moment, third_moment, mean_v_cloud, v_width_cloud, N_cloud)

        except (ValueError, ZeroDivisionError, OverflowError) as e:
            print(f"Invalid parameter theta = {theta}")
            print(f"Error: {e}")
            return small_num, (first_moment, second_moment, third_moment, mean_v_cloud, v_width_cloud, N_cloud)

#2.2 This is to define the parameter range for each one in theta. 
#Walkers will only be allowed to walk inside the range.
def log_prior(theta):
    eta_M, eta_M_cold_tot, log_M_cloud0 = theta
    
    if 0.03 < eta_M < 30.0 and 0.03 < eta_M_cold_tot < 30.0 and 1.0 < log_M_cloud0 < 9.0:
        return 0.0
    else:
        #print(f"invalid log_prior: theta = {theta} is out of bounds!")
        return -np.inf


#2.3 This is what emcee calls first.
def log_probability(theta, x, y, yerr, galaxy_properties, Method):
    small_num = -np.inf

    lp = log_prior(theta)
    if not np.isfinite(lp):
        # Return log_prob and six NaNs matching blobs_dtype
        return small_num, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    log_like, blob = log_likelihood(theta, x, y, yerr, galaxy_properties, Method)

    if not np.isfinite(log_like):
        # If likelihood fails, still return correct blob format
        return small_num, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Unpack the six blob values
    first_moment, second_moment, third_moment, mean_v_cloud, v_width_cloud, N_cloud = blob

    return lp + log_like, first_moment, second_moment, third_moment, mean_v_cloud, v_width_cloud, N_cloud
    


#3. Main code starts here.
#3. Main code starts here.
#3. Main code starts here.

#3.0.1 Define some switches

ifDebug  = 1
ifMCMC   = 0            # = 0 will not run MCMC fit, but read previous saved results
ifPlotCLASSYIII = 1    # Overlay the CLASSY dN/dV versus the model - not valid for the server since I don't upload the PISolution.sav

WhichObj = 'J0021+0052' #='ALL' for all objects, ='TEST' for a test case (fake data), = 'any single object name' to run one object
                        #'J0055-0021', J0021+0052
WhichMod = '001_FitNH'  #Note this will put the results into different folders!
ifShowFig= 0            #If you run all objects, it is better to not showing any figures - figures are always saved to files.


# Manual override for galaxy properties (None = use from the best-fit results)
SFR_manual  = None    # Msun/yr
r0_manual   = None    # kpc
vcirc_manual= None    # km/s

manual_prior = { #Note the prior is eta_M_hot, eta_M_cold, and log_M_cl
    'J0144+0453': [1.0, 1.0, 6.0],
    'J0940+2935': [1.0, 1.0, 6.0],
    'J0944-0038': [1.0, 1.0, 6.0],
    'J1016+3754': [1.0, 1.0, 6.0],
    'J1105+4444': [1.0, 1.0, 6.0],
    'J1112+5503': [1.0, 1.0, 6.0],
    'J1119+5130': [1.0, 1.0, 6.0],
    'J1129+2034': [2.0, 1.0, 6.0],
    'J1150+1501': [1.0, 1.0, 6.0],
    'J1225+6109': [1.0, 1.0, 6.0],
    'J1314+3452': [1.0, 1.0, 6.0],
    'J1359+5726': [1.0, 1.0, 6.0],
    'J1545+0858': [1.0, 1.0, 6.0],
    'J1418+2102': [1.0, 1.0, 6.0],
    'J1444+4237': [3.0, 3.0, 6.0],
}



#If you rerun the fit, I suggest to add the blobs to store extra parameters to the MCMC chain.
#if ifMCMC == 1:
    #https://emcee.readthedocs.io/en/stable/user/blobs/
#    raw

#001 - 003 is the methods to fit Mdot, pdot, Edot for each galaxy. Now pdot and Edot have higher orders of v terms, so they actually trace the shapes of dN/dV differently than Mdot
#001) half_opening_angle = np.pi/2; Mdot_coefficient = 1/3;
#002) half_opening_angle = np.pi/4; Mdot_coefficient = 1/3;
#003) half_opening_angle = np.pi/2; Mdot_coefficient = 1

#001_FitNH is the methods to fit NH, vcenter, and sigma_v instead
#001_FitNH) half_opening_angle = np.pi/2; Mdot_coefficient = 1/3;


#Notes this Omwind parameter is now not used! This is because I'm fitting the term without Omwind*mu*mp*r for CLASSY galaxies.

#These should match the ones in Multiphase_wind_fitting_function.py
#I use them to tune the observed values in ReadObservations (note half_opening_angle = np.pi/2 for CLASSY III paper - i.e., Omwind = 4pi)
#todos: I need to move these also as input parameter to Multiphase_wind_fitting_function.py
#half_opening_angle = np.pi/4       #Cor M82 we have open angle = np.pi*36.85/180.0
#Omwind             = 4*np.pi*(1.0 - np.cos(half_opening_angle))
#if ifDebug == 1:
#    print(f"Omwind = {Omwind}") # = 3.68 rad for pi/4 opening angle (v.s. 12.56 for pi/2)



#3.0.2 Read some files
#3.0.2.1 Note some of the object names contain /, so I need to fix
#m1 - m3 are for outflow rates without the constant (i.e., three moments of dN/dV)
#N1 - N3 are for v, sigma_v, and NH
MasterPATH = './' #Note CLASSY files copied from '/Applications/Work-Desktop/CLASSY/Analysis/PlotSiII/'
objname, zobj, m1, m1_err, m2, m2_err, m3, m3_err,  \
    N1, N1_err, N2, N2_err, N3, N3_err, Int_left, Int_rigt, \
    r50, sfr, vcir,  \
    Mdot, Pdot, Edot, MdotErrUp, MdotErrDo, PdotErrUp, PdotErrDo, EdotErrUp, EdotErrDo = ReadObservations(MasterPATH)

#print(f"objname = {objname}")
#print(f"Int_left = {Int_left}")
#print(f"Int_rigt = {Int_rigt}")


#This is required to multi-processing, i.e., only allow the code to be called in directly in terminal.
if __name__ == '__main__':
    
    for obji in range(0,len(objname)):
    #for obji in range(7,8):
        
        #3.1 Get the correct objectname
        oneobjname = objname[obji].lstrip()
        
        if WhichObj == 'ALL':
            a = 0 #placeholder
        elif WhichObj == 'TEST':
            if obji == 0: #only run test once.
                oneobjname = 'TEST'
            else:
                break
        elif WhichObj != objname[obji]: #here you need to match a specific object
            continue
            
        Method = extract_method(WhichMod)

        print(f"Start fitting object = {oneobjname} with zobj = {zobj[obji]}, obji = {obji}, method = {Method}")
  
        #3.1.2
        folder_path = "./Results/Model_"+WhichMod+"/"+oneobjname+"/"
        os.makedirs(folder_path, exist_ok=True)

        #3.2 Define the moments to be fitted
        if Method == None: #for model 001 - 003 I don't add the suffix. These are the models to fit the three moments
            if WhichObj == 'TEST': #Note y and yerr values are in linear scale here
                x    = np.array([1.0,2.0,3.0]) #placeholder
                y    = np.array([9.10e+28, 5.6e+36, 3.7e+44]) #This is just a test and here are the fake 1st, 2nd, 3rd moments.
                yerr = np.array([0.1e+28, 0.1e+36, 0.1e+44])
            else:
                x    = np.array([1.0,2.0,3.0])
                y    = np.array([m1[obji], m2[obji], m3[obji]]) 
                yerr = np.array([m1_err[obji], m2_err[obji], m3_err[obji]]) 
                
        elif Method == 'FitNH':    #Note y and yerr values are in log scale here
            if WhichObj == 'TEST':
                x    = np.array([1.0,2.0,3.0])
                y    = np.array([7., 7.3, 20.50]) #This is just a test and here are the fake 1st, 2nd, 3rd moments.
                yerr = np.array([0.1, 0.15, 0.3])
            else:
                x    = np.array([1.0,2.0,3.0])
                y    = np.array([N1[obji], N2[obji], N3[obji]])
                yerr = np.array([N1_err[obji], N2_err[obji], N3_err[obji]])             

        print(f"input x    = {x},   type = {np.array(x).dtype}")
        print(f"input y    = {y},   type = {np.array(y).dtype}")
        print(f"input yerr = {yerr},type = {np.array(yerr).dtype}")

        
        #sanity check 
        if np.array_equal(y, [None, None, None]):
            print("This object does not have outflows, skip!")
            continue


        
        #3.3 Fixed parameters that passed to emcee (different for each object)
        #Note these units are defined in Multiphase_Wind_Fitting_function.py
        SFR_value = sfr[obji] if SFR_manual is None else SFR_manual
        r0_value  = r50[obji] if r0_manual is None else r0_manual
        vcirc_value = vcir[obji] if vcirc_manual is None else vcirc_manual

        SFR             = SFR_value * Msun/yr
        r0              = r0_value * 1000 * pc #Note this is the r50 from CLASSY III papers (table A2). And my outflow rates there was using R = 2*r50
        v_circ0         = vcirc_value * km/s

        Z_cloud_initial = 1 # relative to Z_solar
        eta_E           = 1
        NH_int_left     = Int_left[obji]
        NH_int_rigt     = Int_rigt[obji]

        galaxy_properties = np.zeros(7)
        galaxy_properties[0] = SFR
        galaxy_properties[1] = r0
        galaxy_properties[2] = v_circ0
        galaxy_properties[3] = Z_cloud_initial
        galaxy_properties[4] = eta_E
        galaxy_properties[5] = NH_int_left
        galaxy_properties[6] = NH_int_rigt
        

        print("=== Galaxy Properties ===")
        print(f"SFR (Msun/yr)          = {galaxy_properties[0]/(Msun/yr):.3f}")
        print(f"r0 (kpc)                = {galaxy_properties[1]/pc/1000:.3f}")
        print(f"v_circ0 (km/s)         = {galaxy_properties[2]/(km/s):.3f}")
        print(f"Z_cloud_initial (Zsun) = {galaxy_properties[3]:.3f}")
        print(f"eta_E                  = {galaxy_properties[4]:.3f}")
        print(f"NH_int_left (km/s)     = {galaxy_properties[5]:.3f}")
        print(f"NH_int_right (km/s)    = {galaxy_properties[6]:.3f}")
        print("=========================")
        
        #print(f"SFR, rstar, v_cir = {sfr[obji], r50[obji], vcir[obji]}")
        #print(f"SFR, rstar, v_cir with units = {SFR, r0, v_circ0}")
        #print(f"Mdot, Pdot, Edot = {Mdot[obji], Pdot[obji], Edot[obji]}")

        
        #3.4 Define the settings of the fit
        #eta_M              initial hot phase or single phase mass loading
        #eta_M_cold_tot     initial cold phase mass loading, called eta_M_cold previously
        #log_M_cloud0       initial cloud mass, called log_M_cloud_init previously
        
        data = (x, y, yerr, galaxy_properties, Method)
        nwalkers =64
        niter = 256
        initial = np.array([Mdot[obji]/sfr[obji], Mdot[obji]/sfr[obji], 5]) #These are the initial values for the three variables, i.e., eta_M, eta_M_cold_tot, log_M_cloud0. 
                                          #Need to be consistent with log_likelihood().
        ndim = len(initial)
        print(f"initial params without correction = {initial}")

        #Check the initial values are within the prior ranges and output it.
        # 3.4.2 Override if object has a manual prior
        if oneobjname in manual_prior:
            initial = np.array(manual_prior[oneobjname])

        #3.4.3 Check if prior is correct:
        if log_prior(initial) != 0.0:
            print("Warning: the initial params are out of the range! Please specify them manually here!")

            if initial[0] > 10.: #adjust this accordingly if Mdot is too large or too small. Note this criteria has to be outside the range of log_prior function
                initial = np.array([1.0, 1.0, 6])
            elif initial[0] < 0.01:
                initial = np.array([0.2, 0.2, 6]) 
            else:    #otherwise, M_cl may be out of range??
                raw()
            print(f"updated initial params = {initial}")

            # Double-check that the new initial is valid
            if log_prior(initial) != 0.0:
                raise ValueError("Initial parameter adjustment failed! Still wrong prior and you need to fix!")

        
        print(f"initial params after correction = {initial}")
        
        #3.5 For each walker, create their initial position (i.e., theta)
        delta_pos = np.array([0.5, 0.5, 1.0])                  # Perturbation sizes for each parameter in the initial position
        p0 = [np.array(initial) + delta_pos * np.random.randn(ndim) for i in range(nwalkers)] 
        
        # Print initial positions for verification
        if ifDebug == 1:
            print("Initial positions of walkers (p0):")
            for i, pos in enumerate(p0):
                if i <=5:
                    print(f"Walker {i}: {pos}")
                else:
                    break #dont output too much
        
        #3.6 Set up the backend. Don't forget to clear it in case the file already exists
        bkfilename = folder_path+"backend.h5"

        #4. Run in parallel
        if ifMCMC == 1:
            backend = emcee.backends.HDFBackend(bkfilename)
            backend.reset(nwalkers, ndim)    

            # Define the blob dtype to store extra outputs from the likelihood
            blobs_dtype = [("first_moment", float),
                           ("second_moment", float),
                           ("third_moment", float),
                           ("mean_v_cloud", float),
                           ("v_width_cloud", float),
                           ("N_cloud", float)]

            with Pool() as pool:

                #4.0 Define the emcee main pro
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                                args=data,
                                                pool=pool,
                                                backend=backend,
                                                blobs_dtype=blobs_dtype)
                
                #4.1 get the workers spread out into different positions to initialize them.
                print("Running burn-in...")
                state = sampler.run_mcmc(p0, 10, progress=True, store=True)
                sampler.reset()

                #4.2 Run full iterations and record the time
                start = time.time()

                print("Running production...")
                sampler.run_mcmc(state, niter, progress=True, store=True)

                end = time.time()
                multi_time = end - start

                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        else:
            sampler = -1 #placeholder

        
        #5. Plot the best model and the corner plots
        MakePlots(oneobjname, sampler, folder_path, x, y, yerr, initial, galaxy_properties,\
                  Mdot[obji], Pdot[obji], Edot[obji], MdotErrUp[obji], MdotErrDo[obji], PdotErrUp[obji], \
                  PdotErrDo[obji], EdotErrUp[obji], EdotErrDo[obji], ifDebug, Method, ifPlotCLASSYIII,ifShowFig)

        #6. Plot the radial distribution all different parameters
        #This has been moved to Plot_Results.ipynb


