"""
MODULE: sampler
@author: Benjamin Pieczynski
DATE: 2024-06-03

PURPOSE:
    Perform sampling operations on the DataFrames.
    
INCLUDED FUNCTIONS:
    random_oversampler
    
MODIFICATION HISTORY:
    NONE
"""

# imports
import random
import pandas as pd
import numpy as np

def random_oversampler(X_temp: pd.DataFrame, y_temp: pd.DataFrame, adjustment: int) -> tuple:
    """
    Randomly over-sample underrepresented data using a suggested bin

    Parameters
    ----------
    X_train (pd.DataFrame): training data
    y_train (pd.DataFrame): result values to train on
    adjustment (int): contains suggested number of adjustment
    
    Returns
    -------
    X_temp_rs (pd.DataFrame): resampled temporary data array
    y_temp_rs (pd.DataFrame): resampled temporary data array
    """

    # iterate through each bin
    dup_indices = random.sample(list(X_temp.index.values), adjustment)
    
    # grab rows to duplicate
    dup_X = X_temp.loc[dup_indices]
    dup_y = y_temp.loc[dup_indices]

    # concatenate all the df arrays into a single DataFrame and series
    X_temp_rs = pd.concat([X_temp, dup_X], ignore_index=True, axis=0)
    y_temp_rs = pd.concat([y_temp, dup_y], ignore_index=True, axis=0)

    return X_temp_rs, y_temp_rs


def random_undersampler(X_temp: pd.DataFrame, y_temp: pd.DataFrame, adjustment: int):
    """
    Randomly under-sample underrepresented data using a suggested bin

    Parameters
    ----------
    X_train (pd.DataFrame): training data
    y_train (pd.DataFrame): result values to train on
    adjustment (pd.DataFrame): number to sample and drop
    
    Returns
    -------
    X_temp_rs (pd.DataFrame): resampled data array
    y_temp_rs (pd.DataFrame): resampled data array
    """

    # get  random sample of indices to remove
    drop_indices = random.sample(list(X_temp.index.values), adjustment)

    # remove indices from the data set
    X_temp_rs = X_temp.drop(drop_indices, axis=0)
    y_temp_rs = y_temp.drop(drop_indices, axis=0)
    
    return X_temp_rs, y_temp_rs

    
def resampler(X_train: pd.DataFrame, y_train: pd.DataFrame, suggested_df: pd.DataFrame):
    """
    Used to resample data, through over / under sampling

    Parameters
    ----------
    X_train (pd.DataFrame): training data
    y_train (pd.DataFrame): result values to train on
    suggested_df (pd.DataFrame): contains suggested training method and adjustments
    
    Returns
    -------
    X_resampled (pd.DataFrame): resampled data array
    y_resampled (pd.DataFrame): resampled data array
    """

    # temporary arrays to build resampled df
    temp_X_arr = []
    temp_y_arr = []
    
    # iterate through each bin
    for _, row in suggested_df.iterrows():
        bin_num = row['bin#']
        n_adjust = int(row['adjustment'])
        resample_option = row['resample']
        w = y_train['bin#'] == bin_num
        X_temp = X_train[w]
        y_temp = y_train[w]

        try:
            # oversampling
            if resample_option == 'over':
                print('Randomly Oversampling - bin#{}'.format(row['bin#']))
                X_temp_rs, y_temp_rs = random_oversampler(X_temp, y_temp, n_adjust)
                print(len(y_temp), '+ >>', len(y_temp_rs))

            # under-sampling
            elif resample_option == 'under':
                print('Randomly Under-sampling - bin#{}'.format(row['bin#']))
                X_temp_rs, y_temp_rs = random_undersampler(X_temp, y_temp, n_adjust)
                print(len(y_temp), '- <<', len(y_temp_rs))

            else:
                print('no resampling')

            # append the filtered bin to the new array
            temp_X_arr.append(X_temp_rs)
            temp_y_arr.append(y_temp_rs)
        
        except:
            print('No samples from bin# in train-test split')

    # concatenate all the df arrays into a single DataFrame and series
    X_resampled = pd.concat(temp_X_arr, ignore_index=True, axis=0)
    y_resampled = pd.concat(temp_y_arr, ignore_index=True, axis=0)
    
    return X_resampled, y_resampled
    