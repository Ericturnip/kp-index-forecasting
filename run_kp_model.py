"""
MODULE: run_kp_model
--------------------
@author: Benjamin Pieczynski
DATE: 2024-06-11

PURPOSE:
    Run the different versions of Kp models
    
    Linear Model
    ------------
    Kp = A + B*dphi_dt + C*V^2*sqrt(N) + D*10^(0.5-bz) + C*10^(1 + (dbz / dt))
    where dphi_dt = V^(4/3)*b_t^(2/3)*(np.sin(theta_c / 2))^(8/3)

INCLUDES:
    kp_model
    linear_model
    run_linear_kp_model
    
MODIFICATION HISTORY:
    NONE
"""

# imports
import numpy as np
import pandas as pd

# user imports
from model_analysis import test_linear_model, plot_kp_analysis
from utils import df_index_time2yrfrac, df_derivative
from parabolic_fit import kp_amplifier_dip_finder

def kp_model_arrays(df: pd.DataFrame, vopt: int = 1) -> tuple:
    """_summary_

    parameters
    ----------
        df (pd.DataFrame): Contains data for all terms 
        vopt (int, optional): selected viscous term. Defaults to 1.

    raises
    ------
        ValueError: for incorrect vopt

    returns
    -------
        coupling_func (np.ndarray): coupling function value
        viscous (np.ndarray): viscous terms
    """
    
    print('\nComputing model terms...')
    theta_c = (np.arctan2(df['by'], df['bz']) + 2*np.pi) % (2*np.pi)
    b_t = np.sqrt(df['bx']**2 + df['by']**2 + df['bz']**2)

    #dphi_dt
    coupling_func = df['velocity']**(4/3)*b_t**(2/3)*(np.sin(theta_c / 2))**(8/3)
    
    # set initial values
    bz_term = 10**(0.5 - df['bz'])
    #dbzdt_term = 10**(1 - df_derivative(df, 'bz'))
    dbzdt_term = kp_amplifier_dip_finder(df)
    #print(len(bz_term), len(dbzdt_term))
    
    # viscous term
    if vopt == 1:
        viscous = df['density']**0.5 * df['velocity']**2
    elif vopt == 2:
        viscous = df['density']**0.5 * df['velocity']**3
    elif vopt == 3:
        pass # spot for a new option
    else:
        raise ValueError(f'vopt must be in [1,2,3]')

    # create a temporary DataFrame
    temp_df = pd.DataFrame({
        'coupling_term': coupling_func,
        'viscous_term': viscous,
        'bz_term': bz_term,
        'dbzdt_term': dbzdt_term,
        'bz': df['bz']
    })

    # set terms to be 0 if CME condition not met
    temp_df.loc[temp_df['bz'] >= -0.5, ['dbzdt_term', 'bz_term']] = 0

    #for col in temp_df.columns.tolist():
    #    print(type(temp_df[col]), len(temp_df[col]))

    return temp_df['coupling_term'], temp_df['viscous_term'], temp_df['bz_term'], temp_df['dbzdt_term']

def kp_model(df: pd.DataFrame,
             A: float = 0.5,
             B: float = 2.244e-4,
             C: float = 2.844e-6,
             D: float = 0,
             E: float = 0,
             amp: float = 1) -> tuple:
    """
    Linear Model for Kp Index

    Parameters:
    -----------
    A (float): y-intercept
    B (float): coupling term
    C (float): viscous term
    D (float): amplitude multiplier (not-trained)
    df (pd.DataFrame): loaded data
    
    Returns:
    --------
    y_pred (np.ndarray): kp model predictions
    y_test (np.ndarray): kp actual values
    """

    # model components
    theta_c = (np.arctan2(df['by'], df['bz']) + 2*np.pi) % (2*np.pi)
    b_t = np.sqrt(df['bx']**2 + df['by']**2 + df['bz']**2)
    dphi_dt = df['velocity']**(4/3)*b_t**(2/3)*(np.sin(theta_c / 2))**(8/3)
    viscous = np.sqrt(df['density']) * df['velocity']**2

    # new model components
    bz_term = []
    #dbzdt_term = []
    for _, row in df.iterrows():
        if row['bz'] <= -0.5:
            bz = 10**(0.5 - df['bz'])
            #dbzdt = 10**(1 - df_derivative(df, 'bz'))
        else:
            bz = 0
            #dbzdt = 0
            
        bz_term.append(bz)
        #dbzdt_term.append(dbzdt)
    
    dbzdt = kp_amplifier_dip_finder(df)
    
    # model result
    y_pred = []
    kp_pred = A \
              + (B*dphi_dt \
              + C*viscous \
              + D*bz_term \
              + E*dbzdt) \
              * amp

    y_pred = np.array(y_pred)
    y_test = np.array(df['kp'])

    return y_pred, y_test


def run_linear_kp_model(params: tuple, old_params: tuple, 
                        X_test: pd.DataFrame, y_test: np.ndarray, old_df: pd.DataFrame,
                        out_dir: str, kp_df: pd.DataFrame, plotting_params=('0', True, 1, None)) -> tuple:
    """
    Run the linear model on a dataset using the 3 coefficients
    
    Parameters:
    -----------
        params (tuple): contains parameters in order (A, B, C, D, E, amp)
        old_params (tuple): similar to parameters, but with older ones
        X_test (pd.DataFrame): testing data
        y_test (np.ndarray): testing data (results)
        old_df (pd.DataFrame): non-features DataFrame
        out_dir (str): path containing output directory
        window_size (int): size of smoothing window in days
        bPlot (bool): T/F on plotting result
    """
    
    # save plotting parameters
    time_str, bPlot, window_size, forecast_time = plotting_params
    
    # store parameters
    A0, B0, C0, D0, E0, amp0 = old_params
    A, B, C, D, E, amp = params
    
    # add models to df
    old_kp = A0 \
             + (B0*X_test['coupling_term'] \
             + C0*X_test['viscous_term'] \
             + D0*X_test['bz_term'] \
             + E0*X_test['dbzdt_term']) \
             * amp0
    new_kp = A \
             + (B*X_test['coupling_term'] \
             + C*X_test['viscous_term'] \
             + D*X_test['bz_term'] \
             + E*X_test['dbzdt_term']) \
             * amp
    
    # test model in new time frame
    print('\nNEW MODEL')
    mae, rmse, corr = test_linear_model((A, B, C, D, E, amp), 
                                        X_test, y_test)
    print('\nOLD MODEL')
    test_linear_model((A0, B0, C0, D0, E0, amp0), X_test, y_test)
    
    # Create a date column for time-series
    indices = X_test.index
    date_array =  df_index_time2yrfrac(old_df, indices)
    
    indices = kp_df.index
    kp_df['date'] = df_index_time2yrfrac(kp_df, indices)
    
    # plot
    df = pd.DataFrame({
        'date': date_array,
        'kp': y_test.values,
        'old_kp_model': old_kp,
        'new_kp_model': new_kp,
        'bz': old_df['bz'],
        'velocity': old_df['velocity'],
        'density': old_df['density']
    })

    if bPlot:
        plot_kp_analysis(df, kp_df, out_dir, time_str, 
                         window_size=window_size, 
                         forecast_time=forecast_time)
    
    return mae, rmse, corr
