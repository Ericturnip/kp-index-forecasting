"""
MODULE: parabolic_fit
@author: Benjamin Pieczynski
DATE: 2024-07-19

PURPOSE:
    Provide functions that can fit parabolas within a time-series

INCLUDES:
    window_sign_slicing
    parabola_step
    find_dips
    kp_amplifier_dip_finder
    
MODIFICATION HISTORY:
    NONE
"""

# imports
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# user imports
import utils

### FUNCTIONS ###
def window_sign_slicing(dy, ddy, window: int, 
                        i: int, step: int) -> tuple:
    """
    Slices arrays to find the mean and signs for dy and ddy
    
    Parameters:
    -----------
        dy (array): numpy or list of first derivative
        ddy (array): numpy or list of second derivative
        window (int): size of the window to compute a mean for
        i (int): starting index
        step (int): direction of stepping
    
    Returns:
    --------
        (tuple): (dy_mean, sign_dy_mean, ddy_mean, sign_ddy_mean)
    """
    for n in range(window, 0, -1):
        slice_0 = i - n + 1 if step == -1 else i
        slice_1 = i + 1 if step == -1 else i + n
        try:
            dy_mean  = np.mean(dy[slice_0:slice_1])
            ddy_mean = np.mean(ddy[slice_0:slice_1])
            sign_dy_mean  = np.mean(np.sign(dy[slice_0:slice_1]))
            sign_ddy_mean = np.mean(np.sign(ddy[slice_0:slice_1]))
            break
        except:
            pass

    return (dy_mean, sign_dy_mean, ddy_mean, sign_ddy_mean)


def parabola_step(df, dy, ddy,
                 start_idx, step, window=3,
                 zero_tol=0.02):
    """
    Iterates through data to find locations where there might be a dip edge.

    Parameters:
    -----------
        df (pd.DataFrame): data
        dy (array): numpy or list containing first derivatives
        ddy (array): numpy or list containing second derivatives
        start_idx (int): starting index
        step (str / int): in 'forward', 'f', 1 or 'backward', 'b', -1
        window (int, optional): forward / backward steps to take average
        zero_tol (float, optional): Tolerance for 0 slope. Defaults to 0.02.

    Returns:
    --------
        int: edge index
    """
    
    if step in ['forward', 'f', 1]:
        step = 1
        sign_dy = 1
    elif step == ['backwarward', 'b', -1]:
        step = -1
        sign_dy = -1
    
    sign_dy = step
        
    idx_array = df.index
    i = start_idx
    stop_idx = max(idx_array) if step == 1 else min(idx_array)
    
    # slope switches
    dy0_mean, _, ddy0_mean, _ = window_sign_slicing(dy, ddy, window, start_idx, step)
    dy_st = dy[start_idx]
    ddy_st = ddy[start_idx]
    
    while i != stop_idx:
        dy_mean, sign_dy_mean, ddy_mean, sign_ddy_mean = window_sign_slicing(dy, ddy, window, i, step)
                
        # conditions to end the loop
        if dy_mean < 0 + zero_tol and dy_mean > 0 - zero_tol:
            if ddy_mean < 0 + zero_tol and ddy_mean > 0 - zero_tol:
                return i

        if sign_dy_mean != sign_dy:
                if abs((ddy_mean - ddy0_mean) / ddy0_mean) <= 1.1:
                    pass
                else:
                    return i
        
        # increment steps
        if step == 1:
            i += 1
        else:
            i -= 1
        return i


def find_dips(df, val_col: str, time_col: str, 
              max_minima: float, zero_tol=0.02) -> tuple:
    """
    Locates dip edges using above functions.

    Parameters:
    -----------
        df (pd.DataFrame): data
        val_col (str): column for value
        time_col (str): column for time
        max_minima (float): threshold value to look for dips
        zero_tol (float, optional): zero slope tolerance
        
    Returns:
    --------
        tuple: (edge_times, edge_values)
    """
    # smooth data frame
    search_df = utils.boxcar_time_smoothing(df, 1,
                                            time_col, val_col)
    
    # set t and y variables
    y = search_df[val_col]
    t = search_df[time_col]
    
    # take derivatives
    dy  = np.gradient(y, t)
    ddy = np.gradient(dy, t)
    
    # build gradient DataFrane
    grad_df = pd.DataFrame({
        't':   t,
        'y':   y,
        'dy':  dy,
        'ddy': ddy
    })
    
    # save indices from the gradient DataFrame
    index_array = grad_df.index
    
    # filter gradient DataFrame for max minima
    filtered_df = grad_df[grad_df['y'] <= max_minima]
    print(filtered_df)
    # find peaks in inverted filtered data\
    inverted_y = -filtered_df['y']
    peaks, _ = find_peaks(inverted_y, prominence=0.4)

    # extract minima
    min_indices = filtered_df.index[peaks]
    
    # check edge cases at the beginning and end of the data
    max_idx = np.min(index_array)
    min_idx = np.max(index_array)

    if min_idx not in min_indices:
        row = grad_df.iloc[max_idx]
        if row['y'] <= max_minima and row['dy'] < zero_tol:
            min_indices = np.append(min_indices, min_idx)

    if max_idx not in min_indices:
        row = grad_df.iloc[max_idx]
        if row['y'] <= max_minima and row['dy'] < zero_tol:
            min_indices = np.append(min_indices, max_idx)
    
    # save minima
    #minima = grad_df.loc[min_indices]

    # find edges
    edge_indices = []
    for i in min_indices:
        start_idx = parabola_step(grad_df, grad_df['dy'], grad_df['ddy'], i, step=-1)
        stop_idx  = parabola_step(grad_df, grad_df['dy'], grad_df['ddy'], i, step= 1)
        edge_indices.append(start_idx)
        edge_indices.append(stop_idx )
    
    # save edge indices into a DataFrame
    try:
        edges = grad_df.loc[edge_indices]
    except:
        repaired_edges = []
        n = 0
        for i in range(0, len(edge_indices), 2):
            pair = (edge_indices[i], edge_indices[i+1])
            if None in pair:
                None
            else:
                repaired_edges.append(pair[0])
                repaired_edges.append(pair[1])
        edges = grad_df.loc[repaired_edges]
            
    
    # save edges as times and values
    edge_times = edges['t'].values.astype(float)
    edge_vals  = edges['y'].values.astype(float)
    
    return edge_times, edge_vals


def kp_amplifier_dip_finder(df: pd.DataFrame) -> list:
    """
    Provides a 4th term to amplify Kp values. The amplifier is included at
    points that are a part of a dip below a specific max_minima of -0.5.

    Parameters:
    -----------
        df (pd.DataFrame): data

    Returns:
    --------
        list: array of amplified values
    """
    # compute fractional year for time keeping
    frac_years = []
    for _, row in df.iterrows():
        frac_years.append(utils.fractional_year(row['year'], row['doy'], row['hour']))
    
    df['frac_year'] = frac_years

    # find edge times of dips
    edge_times, edge_vals = find_dips(df, val_col='bz', time_col='frac_year', 
                                      max_minima=-0.5)

    # iterate through each row and set amplified values
    amplifier = []
    for _, row in df.iterrows():
        bDip = False # point within dip
        ct = row['frac_year'] # current time
        for i in range(0, len(edge_times), 2):
            t1 = edge_times[i]
            t2 = edge_times[i+1]
            if ct >= t1 and ct <= t2:
                dip_max_val = np.max((edge_vals[i], edge_vals[i+1]))
                amplifier.append(np.sqrt(np.abs(dip_max_val - row['bz'])))
                bDip = True
                break
        if bDip == False:
            amplifier.append(0)

    return amplifier

