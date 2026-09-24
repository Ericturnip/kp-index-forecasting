"""
MODULE: read_module

@author: Benjamin Pieczynski 2024-05-22

PURPOSE:
    Read in data from e3, bxbybz, and NOAA Kp Index files.

    Read functions intially written by Matthew Bracamontes.

MODIFICATION HISTORY:
    NONE
"""

# imports
import os
import numpy as np
import pandas as pd

# user imports
from utils import days_in_year, dt_converter

def read_kp(kp_file: str, times: list, nskip: int = 1) -> pd.DataFrame:
    
    # store times from times array
    start_time = times[0]
    end_time   = times[1]   

    # arrays to store data
    year     = []
    doy      = []
    hour     = []
    kp_vals  = []
      
    with open(kp_file, 'r') as kp:
        n = 0 # increment lines
        
        for line in kp:
            if n >= nskip:
                line = line.strip().split()
                
                try:
                    # make sure the year and doy is correct
                    cur_yr = int(line[0])
                    yr_days = days_in_year(cur_yr)
                    tdoy = int(line[1])
                    thour = int(float(line[2]))
                    if tdoy > yr_days:
                        tdoy    -= yr_days # day is in the next year
                        cur_yr += 1       # year is one year behind

                    # convert current day into a datetime format to check if it is in range
                    current_time = dt_converter(f'{cur_yr} {tdoy} {thour}', '%Y %j %H')

                    # if time is within range append
                    if current_time >= start_time and current_time <= end_time:
                        kp_vals.append(float(line[6]))
                        year.append(cur_yr)
                        doy.append( tdoy  )
                        hour.append(thour )
                except:
                    year.append(np.nan)
                    doy.append(np.nan)
                    hour.append(np.nan)
                    kp_vals.append(np.nan)
                    print(f'line {n} not parsable')
            n += 1
    
    # build DataFrame
    df = pd.DataFrame({
        'year':      np.array(year),
        'doy':       np.array(doy),
        'hour':      np.array(hour),
        'kp':        np.array(kp_vals),
    })
    df.dropna()

    return df

def read_e3(e3_file: str, times: list, nskip: int = 0) -> pd.DataFrame:
    """
    reads data from e3 files

    parameters
    ----------
        e3_file (str): filename
        times (list): list of datetime objects (start, end)
        nskip (int): number of lines to skip

    returns
    -------
        df: pandas DataFrame
    """
    
    # store times from times array
    start_time = times[0]
    end_time   = times[1]
    
    # arrays to store data
    year     = []
    doy      = []
    hour     = []
    density  = []
    velocity = []
    
    with open(e3_file, 'r') as e3:
        lines = e3.readlines()
        n = 0 # increment lines
        
        for line in lines:
            line = line.strip().split()
            
            try:
                # make sure the year and doy is correct
                cur_yr = int(line[0])
                yr_days = days_in_year(cur_yr)
                tdoy = int(line[1])
                thour = int(float(line[2]))
                if tdoy > yr_days:
                    tdoy    -= yr_days # day is in the next year
                    cur_yr += 1       # year is one year behind

                # convert current day into a datetime format to check if it is in range
                current_time = dt_converter(f'{cur_yr} {tdoy} {thour}', '%Y %j %H')

                # if time is within range append
                if current_time >= start_time and current_time <= end_time and thour in [3, 9, 15, 21]:
                    den = float(line[6])
                    density.append(den)
                    year.append(cur_yr)
                    doy.append( tdoy  )
                    hour.append(thour )
                    try:
                        velo = float(line[7].split('-')[0])
                        velocity.append(velo)
                    except:
                        velo
                        velocity.append(float(line[7]))
                
            except:
                year.append(np.nan)
                doy.append(np.nan)
                hour.append(np.nan)
                density.append(np.nan)
                velocity.append(np.nan)
                print(f'line {n} not parsable')

            n += 1

    # build DataFrame
    df = pd.DataFrame({
        'year':      np.array(year),
        'doy':       np.array(doy),
        'hour':      np.array(hour),
        'density':   np.array(density),
        'velocity':  np.array(velocity),
    })
    df.dropna()

    return df



def read_bxyz(bxyz_file: str, times: list, option: str = 'IPS', nskip: int = 1) -> pd.DataFrame:
    """
    Utilizes pandas to read files from the bxyz files.

    Parameters:
    -----------
        bxyz_file (str): path to bxyz file
        times (list): datetime objects (start, end)
        option (str): 'IPS' or '1AU'
        nskip (int): number of lines to skip

    Returns:
    --------
        df (pd.DataFrame): pandas dataframe containing data
    """
    
    # check selection options
    if option.upper() == 'IPS':
        ix, iy, iz = 1, 2, 3 #IPS indicies
    
    elif option.upper() == '1AU':
        ix, iy, iz = 4, 5, 6 #1AU indicies
    
    else:
        raise ValueError(f"Option must be in ['IPS', '1AU'], not {option}")
    
    # initalize times
    start_time = times[0]
    end_time   = times[1] 
    
    # initialize arrays
    year = []
    doy  = []
    hour = []
    bx   = []
    by   = []
    bz   = []
    
    n = 0 # count lines
    with open(bxyz_file, 'r') as bxyz:
        for line in bxyz:
            if n >= nskip:
                # split segments
                l_seg_1, l_seg_2 = line.split()[0], line.split()
                l_seg_1 = l_seg_1.split(':')

                # append to each array if the time is valid
                try:
                    # make sure the year and doy is correct
                    cur_yr = int(l_seg_1[0])
                    yr_days = days_in_year(cur_yr)
                    tdoy  = int(l_seg_1[1])
                    thour = int(l_seg_1[2])
                    if tdoy > yr_days:
                        tdoy    -= yr_days # day is in the next year
                        cur_yr += 1       # year is one year behind
                    
                    # convert to current time
                    current_time = dt_converter(f'{cur_yr} {tdoy} {thour}', '%Y %j %H')
                    
                    if current_time >= start_time and current_time <= end_time:
                        bx.append(float(l_seg_2[ix]))
                        by.append(float(l_seg_2[iy]))
                        bz.append(float(l_seg_2[iz]))
                        year.append(cur_yr)
                        doy.append(tdoy   )
                        hour.append(thour)

                except:
                    year.append(np.nan)
                    doy.append(np.nan)
                    hour.append(np.nan)
                    bx.append(np.nan)
                    by.append(np.nan)
                    bz.append(np.nan)
                    print(f'line {n} not parsable')
            else:
                pass
            
            n += 1
    
    # build DataFrame
    df = pd.DataFrame({
        'year': np.array(year),
        'doy':  np.array(doy ),
        'hour': np.array(hour),
        'bx':   np.array(bx),
        'by':   np.array(by),
        'bz':   np.array(bz)
    })
    df.dropna()

    return df

def reload_df(df_file: str, times: list) -> pd.DataFrame:
    """
    reloads saved DataFrames from previous attempts
    
    parameters
    ----------
    df_file (str): file to load
    times (list): two element DateTime array [start, end]
    
    returns
    -------
    df (pd.DataFrame): DataFrame with loaded values
    """
    
    # initalize times
    start_time = times[0]
    end_time   = times[1]
    
    # check if the file exists and read it
    if os.path.exists(df_file):
        print('\nReloading previous DataFrame...')
        df = pd.read_csv(df_file, sep=',')
        print(f'reloading: {df_file}... SUCCESS')

        # fix integer types
        df['year'] = df['year'].astype(int)
        df['doy'] = df['doy'].astype(int)
        df['hour'] = df['hour'].astype(int)
        
        #build a date time array
        print('selecting times within range...')
        dt_array = []
        for _, row in df.iterrows():
            
            date_string = '{} {} {}'.format(int(row['year']), 
                                            int(row['doy']), 
                                            int(row['hour']))
            dt_array.append(dt_converter(date_string, '%Y %j %H'))

        # create a truth array
        w = []
        for dt in dt_array:
            if (dt >= start_time) & (dt <= end_time):
                w.append(True)
            else:
                w.append(False)
        
        # filter the times
        df = df[w]

        # remove datetime column
        print('RELOAD SUCCESSFUL\n')
        return df
        
    else:
        raise ValueError(f'{df_file} does not exist\nREAD FAILED')

def load_model(model_path: str) -> tuple:
    """
    loads in a .model file for reuse
    """
    print(f'loading model {model_path}')
    # storage array
    coeff_list = []
    coeff_options = [
        'A',
        'B',
        'C',
        'D',
        'E',
        'amp',
        'bshift'
    ]

    # read the files and store coefficients in a list
    f = open(model_path, 'r')
    lines = f.readlines()

    # linear model loading
    if 'linear' in lines[1].split()[1]:
        for line in lines:
            line = line.split()
            if line[0].strip() in coeff_options:
                coeff_list.append(float(line[1].strip()))
                
        # close return a tuple of the list
        f.close()
        return tuple(coeff_list)
    
    else:
        f.close()
        return


def read_live_params(parm_file: str) -> dict:
    """
    Read in parameters from a live training parameter file.
    
    Parameters:
    -----------
        parm_file (str): path to parameter file
        
    Returns:
    --------
        params (dict): dictionary containing parameters
    """

    params = {}
    f = open(parm_file, 'r')
    for line in f:
        try:
            line = line.split('#')[0]
        except:
            pass
        line = line.split(':')
        if line[0] in ['kp_lo', 'kp_hi']:
            params[line[0]] = float(params[1])
        elif line[0] in ['bResample']:
            params[line[0]] = bool(params[1])
        elif line[0] in ['n_bins']:
            params[line[0]] = int(params[1])
        else:
            params[line[0]] = params[1]
    f.close()

    return

