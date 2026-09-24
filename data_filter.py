"""
MODULE:  data_filter
@author: Benjamin Pieczynski
DATE:    2024-05-28

PURPOSE:
    Provide modules for filtering merged DataFrame data after the
    file read section of the train_kp program.
    
INCLUDED FUNCTIONS:
    filter_and_append
    filter_seasons
    remove_bad_data
    hi_low_filter
    var_time_shift

MODIFICATION HISTORY:
    NONE
"""

# imports
import numpy as np
import pandas as pd
import datetime as datetime

# user imports
from utils import dt_converter
from defaults import season_definitions


def filter_and_append(df: pd.DataFrame, col: str, minval, maxval, filtered_df) -> pd.DataFrame:
    """
    Filters and appends a pandas DataFrame according to columns and values
    
    parameters
    ----------
    df (pd.DataFrame): intial DataFrame
    col (str): column to select data from
    minval (int/float): minimum value
    maxval (int/float): maximum value
    filtered_df (pd.DataFrame): DataFrame to store filtered data
    
    returns
    -------
    filtered_df (pd.DataFrame)
    """

    filtered_rows = df[(df[col] >= minval) & (df[col] <= maxval)]
    filtered_df = filtered_df.append(filtered_rows, ignore_index=True)
    n_added = len(filtered_rows)
    print(f'added {n_added} rows...')

    return filtered_df


def filter_seasons(df: pd.DataFrame, seasons: str) -> pd.DataFrame:
    """
    Filters the DataFrame for different seasons
    
    parameters
    ----------
    df (pd.DataFrame): initial DataFrame
    seasons (str): user string input for seasons '1234'
    
    returns
    -------
    pd.DataFrame: filtered or the same DataFrame
    """

    # all seasons selected
    if len(seasons) == 4:
        return df
    
    # build an empty DataFrame
    filtered_df = df.copy()
    filtered_df = filtered_df[0:0]
    
    for season in seasons:
        if season == '1':
            filtered_df = filter_and_append(df, 'doy', 
                                            season_definitions['1']['start'], 
                                            season_definitions['1']['end'],
                                            filtered_df)
        elif season == '2':
            filtered_df = filter_and_append(df, 'doy', 
                                            season_definitions['2']['start'], 
                                            season_definitions['2']['end'],
                                            filtered_df)
        elif season == '3':
            filtered_df = filter_and_append(df, 'doy', 
                                            season_definitions['3']['start'], 
                                            season_definitions['3']['end'],
                                            filtered_df)
        elif season == '4':
            filtered_df = filter_and_append(df, 'doy', 
                                            season_definitions['4']['start'], 
                                            season_definitions['4']['end'],
                                            filtered_df)
        else:
            raise ValueError(f'Season - {season} not in "1,2,3,4"')

    return filtered_df


def remove_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    parameters
    ----------
    df (pd.DataFrame): initial DataFrame

    returns
    -------
    df (pd.DataFrame): fixed DataFrame
    """
    
    # repair DataFrame
    df.replace(['NaN', 'None'], np.nan, inplace=True)
    df = df.dropna()

    return df


def hi_low_filter(df: pd.DataFrame, kp_high: float, kp_low: float,
                  col: str) -> pd.DataFrame:
    """
    allows for filtering of a DataFrame within a specific range
    
    Parameters
    ----------
        df (pd.DataFrame): data
        kp_high (float): highest possible Kp value
        kp_low (float): lowest possible Kp value
        col (str): column name

    Returns:
        filtered_df (pd.DataFrame): filtered DataFrame
    """
    
    w = (df['kp'] >= kp_low) & (df['kp'] <= kp_high)
    filtered_df = df[w]
    
    return filtered_df


def var_time_shift(df: pd.DataFrame, time_shift: float, shift_cols: list,
                   replace: bool = False) -> pd.DataFrame:
    """
    Shifts a list of df variables to an earlier or later time.
    
    Parameters:
    -----------
    df (pd.DataFrame): data
    time_shift (float): time shift in days
    shift_cols (list): list of columns in the data to shift
    replace (bool): an option to replace a value that would 
                    disappear with the old value.
    
    Returns:
    --------
    pd.DataFrame: shifted data
    """

    # delta
    delta_t = datetime.timedelta(days=time_shift)
    
    # determine maximum number of rows should be searched
    
    # build empty dictionary to build new df
    columns = df.columns
    temp_dict = {col: [] for col in columns}
    
    # create a df copy
    df_copy = df.copy()

    # iterate through each row
    for index, row in df.iterrows():
        targ_time  = dt_converter('{}{}{}'.format(int(row['year']), 
                                                  int(row['doy']), 
                                                  int(row['hour'])), 
                                  '%Y%j%H') \
                                  + delta_t

        # iterate through df_copy
        cond = False
        for search_index, search_row in df_copy.iterrows():
            cur_time = dt_converter('{}{}{}'.format(int(search_row['year']), 
                                                    int(search_row['doy']),
                                                    int(search_row['hour'])), 
                                    '%Y%j%H')
            
            # compare times and add it if there is a match
            if targ_time == cur_time:
                for key in temp_dict:
                    if key in shift_cols:
                        temp_dict[key].append(search_row[key])
                    else:
                        temp_dict[key].append(row[key])
                
                # drop the row from df_copy as necessary
                if time_shift < 0:
                    df_copy = df_copy.drop(index=search_index)
                    
                elif time_shift > 0:
                    df_copy.drop(index=index)
                
                # exit the copy loop
                cond = True
                break

        if cond == False and replace == True:
            for key in temp_dict:
                temp_dict[key].append(search_row[key])

    # combine dictionary into new df
    shifted_df = pd.DataFrame(temp_dict)

    return shifted_df