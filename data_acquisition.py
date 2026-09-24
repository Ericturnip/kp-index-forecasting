"""
MODULE: data_acquisition
@author: Benjamin Pieczynski 
DATE: 2024-05-22

PURPOSE:
    Acquire Kp, velocity, density, bx, by, bz data
    for machine learning purposes.
    
INCLUDED FUNCTIONS:
    build_set_dataframe
    acquire_data
    acquire_reload

MODIFICATION HISTORY:
    NONE
"""

# imports
import os
import pandas as pd
import time
from datetime import datetime, timedelta

# user imports
from defaults import file_formats
from read_module import read_bxyz, read_e3, read_kp, reload_df
from data_filter import filter_seasons, remove_bad_data
import utils

def build_set_dataframe(read_func, search_dir: str, times: list, 
                        archive_set: str) -> pd.DataFrame:
    """
    Set to read in the initial file and build a DataFrame for
    each archive set.

    parameters:
    -----------
        read_func (_type_): function needed for reading file set
        search_dir (str): directory to search for the archive files
        times (list): start_time[0], end_time[1]
        archive_set (str): what file set is being used

    Returns:
        pd.DataFrame: archive DataFrame
    """

    print(f'\nfinding {archive_set} data...')
    df_array = [] # array to store data frames
    files = os.listdir(search_dir)
    
    # build a years list to determine if the file should be searched
    start_year = times[0].year
    end_year   = times[1].year
    year_list = list(range(start_year, end_year +1))
    
    # loop through bxyz files
    for file in files:
        if file.endswith('.txt'):
            for year in year_list:
                fname = file_formats[archive_set.lower()].replace('YYYY', str(year))
                if fname in file:
                    file_path = os.path.join(search_dir, file)
                    print(f'reading {file_path}...')
                    df_array.append(read_func(file_path, times))
                    break
                else:
                    pass
    
    # combine the DataFrames
    print(f'combining {archive_set} DataFrames...')
    combined_df = pd.concat(df_array, ignore_index=True)
    
    # sort the combined DataFrame
    print('sorting and resetting indices...')
    df = combined_df.sort_values(by=['year', 'doy', 'hour'])

    # reset index after sorting
    df.reset_index(drop=True, inplace=True)

    print(f'{archive_set} DataFrame Complete\n')
    
    time.sleep(0.5)
    return df
    
def acquire_data(bxyz_dir: str, e3_dir:str, kp_dir: str,
                 times: list, seasons: str):
    """
    Builds a training and testing file for Kp-Index Machine Learning
    
    parameters
    ----------
        bxyz_dir (str): path to magnetic field data
        e3_dir (str): path to velocity and density data
        kp_dir (str): path to the Kp values for training and testing
        seasons (str): user season input '1234'

    Returns:
    --------
        df (pd.DataFrame): forecast data
        kp_df (pd.DataFrame): Kp data for all time
    """
    
    bxyz_df = build_set_dataframe(read_func=read_bxyz, 
                                  search_dir=bxyz_dir,
                                  times=times, 
                                  archive_set='bxyz')
    e3_df = build_set_dataframe(read_func=read_e3, 
                                search_dir=e3_dir,
                                times=times, 
                                archive_set='e3')
    kp_df = build_set_dataframe(read_func=read_kp, 
                                search_dir=kp_dir,
                                times=times, 
                                archive_set='kp')
    
    # merge all the data frames
    print('\nmerging DataFrames...')
    time.sleep(0.5)
    merged_1_df = pd.merge(bxyz_df, e3_df, on=['year', 'doy', 'hour'])
    df = pd.merge(merged_1_df, kp_df, on=['year', 'doy', 'hour'])
    print('Merge - COMPLETE\n')

    #print(df)
    
    # filter the DataFrame
    print('\nFiltering DataFrame...')
    df = filter_seasons(df, seasons)
    df = remove_bad_data(df)
    print('Filtering COMPLETE\n')
    time.sleep(0.5)
    return df, kp_df

def acquire_reload(df_path: str, times: list, seasons: str) -> pd.DataFrame:
    """
    used to reload and filter data from previously used DataFrames
    
    parameters
    ----------
    df_path (str): path to dataframe file (.txt)
    times (list): two element array [start, end]
    seasons (str): list of seasons to select from ('1234')

    returns
    -------
    pd.DataFrame: reloaded data
    """

    # reload the DataFrame
    print('\nreloading DataFrame...')
    df = reload_df(df_path, times)

    # filter the seasons
    print('Filtering DataFrame...')
    df = filter_seasons(df, seasons)
    df = remove_bad_data(df)
    print('Filtering COMPLETE\n')
    time.sleep(0.5)
    
    return df

def forecast_load(carr_rot, bxyz_dir: str, e3_dir: str, 
                  kp_dir: str):
    """
    Load in data for forecast mode.

    Parameters:
    -----------
        carr_rot (float / str): Carrington Rotation / Forecast time
        bxyz_dir (str): B-field directory
        e3_dir (str): e3 directory
        kp_dir (str): Potsdam Kp directory

    Returns:
    --------
        df (pd.DataFrame): forecast data
        kp_df (pd.DataFrame): Kp data for all time
    """
    bxyz_file = f'{bxyz_dir}/Bxyz_insitu_e3_{carr_rot}'
    e3_file   = f'{e3_dir}/e3_{carr_rot}'
    
    # conver to string
    if type(carr_rot) == 'str':
        carr_rot = float

    td = timedelta(days=100*365)
    car_time = utils.cr_to_date(carr_rot)
    temp_times = [car_time - td, car_time + td]

    # make DataFrames for e3 and Bxyz files
    print('reading files for carr_rot:', carr_rot)
    bxyz_df = read_bxyz(bxyz_file, temp_times)
    e3_df   = read_e3(e3_file, temp_times)
    df_init = pd.merge(bxyz_df, e3_df, on=['year', 'doy', 'hour'])

    # locate stop and end times
    start_row = df_init.iloc[0]
    end_row   = df_init.iloc[-1]
    start_time = utils.dt_converter('{} {} {}'.format(int(start_row['year']), int(start_row['doy']), int(start_row['hour'])),
                                    '%Y %j %H')
    end_time = utils.dt_converter('{} {} {}'.format(int(end_row['year']), int(end_row['doy']), int(end_row['hour'])),
                                    '%Y %j %H')   

    # build Kp data frame
    kp_df = build_set_dataframe(read_func=read_kp, 
                                search_dir=kp_dir,
                                times=[start_time, end_time], 
                                archive_set='kp')

    # merge DataFrames
    df = pd.merge(df_init, kp_df, on=['year', 'doy', 'hour'])
    
    # remove all NaNs
    df = df.dropna()

    # remove all rows with bad velocities
    df = df[df['velocity'] > -9999]
    
    return df, kp_df