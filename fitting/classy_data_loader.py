"""
Data loader for CLASSY galaxy observations.

This module loads and prepares CLASSY galaxy data for fitting with
the multiphase wind model.
"""

import numpy as np
import re
import os


def preprocess_line(line):
    """Replace multiple tabs with single tab."""
    return re.sub(r'\t+', '\t', line)


def parse_errors(error_string):
    """
    Parse error string like '0.5+0.2-0.1' to value and errors.
    
    Parameters
    ----------
    error_string : str
        String containing value and asymmetric errors
        
    Returns
    -------
    tuple
        (value, upper_error, lower_error) or (None, None, None)
    """
    pattern = re.compile(r"\s*([+-]?[0-9]*\.?[0-9]+)([+-][0-9]*\.?[0-9]+)(-([0-9]*\.?[0-9]+))")
    match = pattern.match(error_string)
    
    if match:
        value = float(match.group(1))
        upper_error = float(match.group(2))
        lower_error = float(match.group(4))
        return value, upper_error, lower_error
    else:
        # Try simple float conversion
        try:
            value = float(error_string)
            return value, 0.0, 0.0
        except:
            return None, None, None


def read_column_file(filepath, ncol, nread, delimiter='\t'):
    """
    Read space/tab separated file with specified columns.
    
    Parameters
    ----------
    filepath : str
        Path to data file
    ncol : int
        Total number of columns in file
    nread : int
        Number of columns to read
    delimiter : str
        Column delimiter
        
    Returns
    -------
    tuple
        Tuple of lists for each column
    """
    import csv
    
    with open(filepath, 'r') as f:
        reader = csv.reader((preprocess_line(line) for line in f), 
                          delimiter=delimiter, skipinitialspace=True)
        
        # Initialize storage
        columns = {f'col{i}': [] for i in range(1, nread+1)}
        
        for row in reader:
            if len(row) >= nread and not row[0].startswith('#'):
                for i in range(1, nread+1):
                    columns[f'col{i}'].append(row[i-1])
                    
    return tuple(columns[f'col{i}'] for i in range(1, nread+1))


def clean_galaxy_names(names):
    """
    Clean galaxy names by removing special characters.
    
    Parameters
    ----------
    names : list
        List of galaxy names
        
    Returns
    -------
    list
        Cleaned galaxy names
    """
    cleaned = []
    for name in names:
        # Remove special characters and normalize
        clean = name.split('^')[0].strip()
        clean = clean.replace('--', '-').replace('+ ', '+').replace('- ', '-')
        cleaned.append(clean)
    return cleaned


def load_classy_observations(data_dir=None):
    """
    Load all CLASSY galaxy observations.
    
    Parameters
    ----------
    data_dir : str
        Directory containing CLASSY data files
        
    Returns
    -------
    dict
        Dictionary with galaxy data indexed by galaxy name
    """
    if data_dir is None:
        # Default to Xinfeng_Data in parent directory
        import os
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Xinfeng_Data')
    
    galaxies = {}
    
    # Load galaxy properties (SFR, v_circ, r50)
    props_file = os.path.join(data_dir, 'Out5.3_AncillaryParams.txt')
    names, z, r50_arcsec, r50_kpc, v_circ_str = read_column_file(
        props_file, 7, 5, delimiter='&'
    )
    names = clean_galaxy_names(names)
    
    # Parse v_circ with errors
    v_circ = []
    v_circ_err = []
    for vc_str in v_circ_str:
        val, err_up, err_down = parse_errors(vc_str)
        v_circ.append(val)
        v_circ_err.append((abs(err_up) + abs(err_down)) / 2.0 if val else None)
    
    # Load velocity profiles
    profile_file = os.path.join(data_dir, 'Out10.0_NH_Profile_Info.txt')
    with open(profile_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        
    profiles = {}
    for line in lines:
        if line.strip():
            parts = line.strip().split(',')
            if len(parts) >= 6:
                name = parts[0].strip()
                mean_v = float(parts[1])
                hwhm = float(parts[2])
                log_nh = float(parts[3])
                int_left = float(parts[4])
                int_right = float(parts[5])
                
                profiles[name] = {
                    'mean_v': mean_v,
                    'hwhm': hwhm,
                    'log_nh': log_nh,
                    'int_range': (int_left, int_right)
                }
    
    # Note: SFR would need to be loaded from Excel files
    # For now, use placeholder values that can be updated
    default_sfr = 10.0  # Msun/yr - more typical value
    
    # Combine all data
    for i, name in enumerate(names):
        if name in profiles and profiles[name]['mean_v'] != -100:
            # Valid galaxy with measurements
            profile = profiles[name]
            
            # Convert to log scale for fitting (CGS units)
            log_v = np.log10(abs(profile['mean_v']) * 1e5)      # km/s -> cm/s
            log_width = np.log10(abs(profile['hwhm']) * 1e5)    # km/s -> cm/s
            log_nh = profile['log_nh']                          # Already in log
            
            # Error estimates (20 km/s binning error)
            v_err_linear = 20.0  # km/s
            log_v_err = v_err_linear / (abs(profile['mean_v']) * np.log(10))
            log_width_err = v_err_linear / (abs(profile['hwhm']) * np.log(10))
            log_nh_err = 0.1  # Typical column density error
            
            galaxies[name] = {
                'name': name,
                'redshift': float(z[i]),
                'r50': float(r50_kpc[i]),  # kpc
                'v_circ': v_circ[i],        # km/s
                'v_circ_err': v_circ_err[i],
                'sfr': default_sfr,         # Placeholder
                'log_v': log_v,
                'log_v_err': log_v_err,
                'log_width': log_width,
                'log_width_err': log_width_err,
                'log_NH': log_nh,
                'log_NH_err': log_nh_err,
                'mean_v_kms': profile['mean_v'],
                'hwhm_kms': profile['hwhm'],
                'valid': True
            }
    
    print(f"Loaded {len(galaxies)} valid galaxies out of {len(names)} total")
    return galaxies


def get_galaxy_for_fitting(galaxy_name, galaxies_dict):
    """
    Get galaxy data formatted for fitting.
    
    Parameters
    ----------
    galaxy_name : str
        Name of galaxy
    galaxies_dict : dict
        Dictionary of all galaxy data
        
    Returns
    -------
    dict
        Galaxy data ready for fitting
    """
    if galaxy_name not in galaxies_dict:
        raise ValueError(f"Galaxy {galaxy_name} not found in data")
        
    galaxy = galaxies_dict[galaxy_name]
    
    return {
        'name': galaxy_name,
        'sfr': galaxy['sfr'],
        'v_circ': galaxy['v_circ'],
        'r50': galaxy['r50'],
        'log_v': galaxy['log_v'],
        'log_v_err': galaxy['log_v_err'],
        'log_width': galaxy['log_width'],
        'log_width_err': galaxy['log_width_err'],
        'log_NH': galaxy['log_NH'],
        'log_NH_err': galaxy['log_NH_err']
    }


def update_sfr_values(galaxies_dict, sfr_dict):
    """
    Update SFR values in galaxy dictionary.
    
    Parameters
    ----------
    galaxies_dict : dict
        Dictionary of galaxy data
    sfr_dict : dict
        Dictionary mapping galaxy names to SFR values
    """
    for name, sfr in sfr_dict.items():
        if name in galaxies_dict:
            galaxies_dict[name]['sfr'] = sfr
            

def get_valid_galaxy_list(galaxies_dict):
    """
    Get list of valid galaxies for fitting.
    
    Parameters
    ----------
    galaxies_dict : dict
        Dictionary of all galaxy data
        
    Returns
    -------
    list
        List of valid galaxy names
    """
    return [name for name, data in galaxies_dict.items() if data['valid']]


if __name__ == "__main__":
    # Test loading
    print("Loading CLASSY galaxy data...")
    galaxies = load_classy_observations()
    
    print(f"\nValid galaxies for fitting:")
    valid = get_valid_galaxy_list(galaxies)
    for i, name in enumerate(valid[:5], 1):
        galaxy = galaxies[name]
        print(f"{i}. {name}:")
        print(f"   v_circ = {galaxy['v_circ']:.1f} km/s")
        print(f"   r50 = {galaxy['r50']:.2f} kpc")
        print(f"   log(v) = {galaxy['log_v']:.2f}")
        print(f"   log(width) = {galaxy['log_width']:.2f}")
        print(f"   log(NH) = {galaxy['log_NH']:.2f}")
    
    if len(valid) > 5:
        print(f"... and {len(valid)-5} more galaxies")