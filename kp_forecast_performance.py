#!/home/soft/anaconda3/bin/python3

"""
PROGRAM: kp_forecast_performance
@author: Benjamin Pieczynski
DATE: 2024-07-31

PURPOSE:
    Load in a model and multiple forecast runs to analyze model
    performance for a selected forecast window

MODIFICATION HISTORY:
    None
"""

# imports
import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# user imports
import utils
import forecast_analysis
from arg_handler import coefficient_check
from read_module import load_model
from data_acquisition import forecast_load
from data_filter import var_time_shift
from run_kp_model import kp_model_arrays

def main():

    # argument parser initialization
    prog_name = 'kp_forecast_performance'
    description = '''Test trained model by evaluating their performance
                     over a forecast range and time window.'''
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # help descriptions
    range_help  = '''Sets forecast range. Format as carrington1_carrington2 
                     (e.g. 2282.612_2283.453).'''
    window_help = '''Future window to evaluate forecast in units of hours.'''
    source_help = '''Directory that can be referenced as a source directory
                     to acquire data within sub-directories'''
    kp_dir_help = '''Directory to search for the Kp archive file for the 
                     correct time.'''
    b_dir_help  = '''Directory to locate the forecasted bxyz file.'''
    e3_dir_help = '''Directory to locate the forecasted e3 file.'''
    plot_help   = '''Activate plotting.'''
    model_help  = '''Path to model that will have its coefficients
                     loaded into the program'''
    A_help          = '''Input for y_intercept. Use for testing older models and
                     retraining.'''
    B_help          = '''Input for coupling coefficient. Use for testing older models and
                         retraining.'''
    C_help          = '''Input for viscous coefficient. Use for testing older models and
                         retraining.'''
    D_help          = '''Input for the Bz coefficient. D*10^(0.5 - Bz).'''
    E_help          = '''Input for the dBz / dt coefficient. D*10^(1 - dBz / dt).'''
    amp_help        = '''Amplitude option for increasing the amplitude of
                         the peaks in the model. This is not a trainable
                         parameter.'''
    bshift_help     = '''Shifts the bField in time by X days.'''
    out_dir_help    = '''Path to output directory. Set to current directory by default.'''

    # add in arguments
    parser.add_argument('-fr', '--forecast_range', required=True,
                        help=range_help)
    parser.add_argument('-rlm', '--model', required=True,
                        help=model_help)
    parser.add_argument('-A', metavar='y_intercept', default=0.05,
                        type=float, help=A_help)
    parser.add_argument('-B', metavar='coupling_coefficient', default=2.44e-4, 
                        type=float, help=B_help)
    parser.add_argument('-C', metavar='viscous_coefficient', default=2.844e-6, 
                        type=float, help=C_help)
    parser.add_argument('-D', metavar='bz_coefficient', default=0, 
                        type=float, help=D_help)
    parser.add_argument('-E', metavar='dbzdt_coefficient', default=0, 
                        type=float, help=E_help)
    parser.add_argument('-amp', metavar='amplitude', default=1, 
                        type=float, help=amp_help)
    parser.add_argument('-bshift', '--bshift', type=float,
                        default=0, help=bshift_help)
    parser.add_argument('-w', '--window', required=True,
                        type=float, help=window_help)
    parser.add_argument('-sd', '--source_directory', default=None,
                        help=source_help)
    parser.add_argument('-kpd', '--kp_directory', required=True,
                        help=kp_dir_help)
    parser.add_argument('-bd', '--bxyz_directory', required=True,
                        help=b_dir_help)
    parser.add_argument('-e3d', '--e3_directory', required=True,
                        help=e3_dir_help)
    parser.add_argument('-od', '--output_directory', default=os.getcwd(), 
                        help=out_dir_help)
    parser.add_argument('-plot', '--plot', action='store_true',
                        help=plot_help)
    
    # store arguments in a dictionary
    args = parser.parse_args().__dict__
    
    # store arguments as variables
    try:
        forecast_range = (float(args['forecast_range'].split('_')[0]), # start 
                          float(args['forecast_range'].split('_')[1])) # stop
    except Exception as e:
        print(f'Error with forecast range entry: {e}')

    model_file = args['model']
    window = args['window']
    bPlot = args['plot']
    source_dir = args['source_directory']
    A, B, C, D, E = coefficient_check(args)
    amp = args['amp']
    bshift = args['bshift']
    out_dir = args['output_directory']

    # directory management
    if source_dir:
        kp_dir   = os.path.join(source_dir, args['kp_directory'])
        bxyz_dir = os.path.join(source_dir, args['bxyz_directory'])
        e3_dir   = os.path.join(source_dir, args['e3_directory'])
    else:
        kp_dir   = args['kp_directory']
        bxyz_dir = args['bxyz_directory']
        e3_dir   = args['e3_directory']

    # check if all directories exist
    for directory in [kp_dir, bxyz_dir, e3_dir]:
        if os.path.exists(directory):
            pass
        else:
            raise ValueError(f'PATH not found: {directory}')
    
    # load in model parameters
    if model_file != None:
        A, B, C, D, E, amp, bshift = load_model(model_file)
    #params = A, B, C, D, E, amp
    
    # create lists of files in 
    e3_temp_list   = os.listdir(e3_dir)
    bxyz_temp_list = os.listdir(bxyz_dir)
    if len(e3_temp_list ) == 0 or len(bxyz_temp_list) == 0:
        raise ValueError('No data in forecast directory')
    temp_file_list = e3_temp_list + bxyz_temp_list
    
    # initialize forecast file array
    e3_forecast_files   = []
    bxyz_forecast_files = []

    # filter the lists for the forecast times
    forecast_start_time, forecast_end_time = forecast_range
    for file in temp_file_list:
        file_elements = file.split('_')
        try:
            file_time = float(file_elements[-1])
            file_type = file_elements[0]
        except:
            file_time, file_type = 0, None
        if file_time >= forecast_start_time and file_time <= forecast_end_time:
            if file_type == 'e3':
                e3_forecast_files.append(os.path.join(e3_dir, file))
            elif file_type == 'Bxyz':
                bxyz_forecast_files.append(os.path.join(bxyz_dir, file))
    
    # make sure all times found in each file are found in the other file
    forecast_dict = {} # stores as {time: (e3_idx, bxyz_idx), ...}
    e3_forecast_times   = [file.split('_')[-1] for file in e3_forecast_files]
    bxyz_forecast_times = [file.split('_')[-1] for file in bxyz_forecast_files]
    for i, t1 in enumerate(e3_forecast_times):
        for j, t2 in enumerate(bxyz_forecast_times):
            if t2 == t1:
                forecast_dict[t1] = (i,j)
                break
    
    # sort the forecast dictionary so that we can iterate through time correctly
    forecast_dict = dict(sorted(forecast_dict.items()))
    print('SUCCESS\n')
    #for key in forecast_dict:
    #    e3_idx, bxyz_idx = forecast_dict[key]
    #    print(key, e3_forecast_times[e3_idx], bxyz_forecast_times[bxyz_idx])

    # initialize the forecast and Kp DataFrame
    forecast_df = pd.DataFrame(columns=['date', 'forecast_time', 'kp', 'kp_pred'],
                               dtype=float)
    kp_df = None

    # grab all carrington rotations as floats
    cr_vals = [float(cr) for cr in forecast_dict]

    # loop through each forecast file and select times within the window. Then add them
    # to the DataFrame
    for cr_str, cr in zip(forecast_dict, cr_vals):
        # load in forecast and get forecast time
        df, kp_temp_df = forecast_load(cr_str, bxyz_dir, e3_dir, kp_dir)
        df['frac_year'] = utils.df_index_time2yrfrac(df, df.index)
        df['datetime'] = df['frac_year'].apply(utils.fractional_year_to_datetime)
        forecast_time = df['datetime'].max() - timedelta(days=5)
        end_time = forecast_time + timedelta(hours=window)
        
        # set up kp_temp_df and check if kp_df needs to be concatenated
        kp_temp_df['date'] = utils.df_index_time2yrfrac(kp_temp_df, kp_temp_df.index)
        kp_temp_df['datetime'] = kp_temp_df['date'].apply(utils.fractional_year_to_datetime)
        w = (kp_temp_df['datetime'] > forecast_time) & (kp_temp_df['datetime'] <= end_time)
        kp_temp_df = kp_temp_df[w]
        if kp_df is None:
            kp_df = kp_temp_df
        else:
            kp_df = pd.concat([kp_df, kp_temp_df], ignore_index=True)
            kp_df = kp_df.drop_duplicates(subset=['year', 'doy', 'hour'])
            
        # shift b-field
        if bshift != 0:
            shift_cols = ['bx', 'by', 'bz']
            df = var_time_shift(df, bshift, shift_cols)
            
        # grab model arrays
        dphi_dt, viscous, bz_term, dbzdt = kp_model_arrays(df)
        kp_pred = A \
                  + (B*dphi_dt \
                  + C*viscous \
                  + D*bz_term \
                  + E*dbzdt) \
                  * amp
        df['kp_pred'] = kp_pred # place predictions in DataFrame

        # filter DataFrame
        w = (df['datetime'] > forecast_time) & (df['datetime'] <= end_time)
        df = df[w]

        # concat forecast to forecast_df
        kp_pred = df['kp_pred'].values
        kp_test = df['kp'].values
        date = df['frac_year'].values
        forecast_time = np.array([utils.datetime_to_fractional_year(forecast_time) \
                                  for i in range(len(df))])
        #print(kp_pred.values, '\n', df['kp'].values)
        new_data = pd.DataFrame({
            'date': date,
            'forecast_time': forecast_time,
            'kp': kp_test,
            'kp_pred': kp_pred
        })
        forecast_df = pd.concat([forecast_df, new_data], ignore_index=True)
        
    print(forecast_df)
    time.sleep(2)

    # sort data frames
    forecast_df = forecast_df.sort_values(by='date')
    kp_df = kp_df.sort_values(by='date')
    
    # plot results
    if bPlot:
        #kp_vals = (kp_df['date'].values, kp_df['kp'].values)
        time_str = args['forecast_range']
        forecast_analysis.plot_forecast(forecast_df, out_dir, kp_df, window, time_str)
    return

############### CALL MAIN #################

if __name__ == '__main__':
    main()