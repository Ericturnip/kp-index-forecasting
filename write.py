"""
MODULE: write
@author: Benjamin Pieczynski
DATE:    2024-05-28

PURPOSE:
    Contain all write functions for the purposes of saving program
    output.
    
INCLUDED FUNCTIONS:
    write_df
    write_linear_model

MODIFICATION HISTORY:
    NONE
"""

# imports
import os
import pandas as pd

# user imports
from utils import sci_notate

def write_df(df: pd.DataFrame, outfile: str) -> None:
    """
    Writes a DataFrame file as a csv text file with a
    "," delimeter
    
    parameters
    ----------
    df (pd.DataFrame): collected data
    outfile (str): path of the outfile
    """

    print('\nsaving DataFrame for future session...')
    print(f'saving as: {outfile}')
    
    # check if the outfile exists
    if os.path.exists(outfile):
        print(f'{outfile} already exists...\nWRITE FAILED - no outfile written\n')
        return
    
    # write the outfile
    df.to_csv(outfile, sep=',', index=False)
    if os.path.exists(outfile):
        print('WRITE SUCCESS\n')
    return

def write_linear_model(model_dir: str, date_str: str, model: tuple,
                       mae: float, rmse: float, corr: float,
                       bshift: float = 0.0) -> None:
    """
    Saves coefficients of the linear model
    
    Parameters:
    -----------
    model_dir (str): path to model directory
    date_str (str): train-test split dates
    model (tuple): model coefficients
    mae (float): mean absolute error
    rmse (float): root mean squared error
    corr (float): correlation coefficient
    amp (float): amplitude multiplier

    Returns:
    --------
    Writes a linear_yyyymmddHH.model file
    """
    
    # name the model
    model_name = f'{date_str}'
    
    # determine file name and path
    file_name = f'linear_{model_name}.model'
    file_path = os.path.join(model_dir, file_name)

    # build file data
    A, B, C, D, E, amp = model
    file_data = {
        "name": model_name,
        "model": "linear",
        "A": sci_notate(A),
        "B": sci_notate(B),
        "C": sci_notate(C),
        "D": sci_notate(D),
        "E": sci_notate(E),
        "amp": f"{amp:.4f}",
        "bshift": f"{bshift:.4f}",
        "MAE": f"{mae:.4f}",
        "RMSE": f"{rmse:.4f}",
        "corr": f"{corr:.4f}"
    }
    
    # set widths for alignment
    max_key_width = max(len(key) for key in file_data.keys()) + 4
    max_value_width = max(len(str(value)) for value in file_data.values()) + 2

    # write the file
    print(f"Writing model file: {file_path}...")
    f = open(file_path, 'w+')
    for key, value in file_data.items():
        f.write(f"{key:<{max_key_width}}{value:<{max_value_width}}\n")
    f.close()
    print("SUCCESS")
    return

def write_live_parm(params: dict) -> None:
    """
    Update the live parameter file

    Parameters:
    -----------
        params (dict): dictionary containing parameters

    Returns:
    --------
        None
    """

    content = {
        'e3_dir': params['e3_dir'],
        'b_dir': params['b_dir'],
        'kp_dir': params['kp_dir'],
        'model_prefix': '{} # start of model name'.format(params['model_prefix']),
        'alpha_base': '{} # scalar weighting'.format(params['alpha_base']),
        'kp_lo': '{} # Kp filter - low'.format(params['kp_low']),
        'kp_hi': '{} # Kp filter - high'.format(params['kp_hi']),
        'bResample': '{} # allow training resampling'.format(params['bResample']),
        'n_bins': params['n_bins'],
        'current_year': params['current_year'],
        'spring_equinox': '{} # T/F model built'.format(params['spring_equinox']),
        'vernal_equinox': '{} # T/F model built'.format(params['vernal_equinox']),
        
    }
    
    # write the content to the live_training.parm file
    f = open(os.path.abspath('live_training.parm'), 'w+')
    for key in content:
        f.write('{}: {}\n'.format(key, content[key]))
    f.close()
    return

