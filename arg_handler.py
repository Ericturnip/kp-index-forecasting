"""
MODULE: arg_handler

@author: Benjamin Pieczynski
DATE: 2024-05-23

PURPOSE:
    Handles specific operations with parsing arguments
    that argparse was not designed to handle.
    
INCLUDED FUNCTIONS:
    check_if_int
    time_range_parser
    arg_min_max
    coefficient_check
    check_training

MODIFICATION HISTORY:
    NONE
"""

# imports
import time

# user imports
from utils import dt_converter

def check_if_int(strings: list) -> bool:
    """
    check if a list of strings are all integers

    Parameters
    ----------
        strings (list): list of strings

    Returns
    -------
        bool: True if strings are integers
    """
    for string in strings:
        try:
            int(string)
        except:
            return False
        
    return True


def arg_min_max(str_arg: str, invert: bool = False) -> tuple:
    """
    Takes an argument used to split into a minimum and maximum
    value and stores it as a tuple.

    Parameters
    ----------
        str_arg (str): string argument of format min_max
        invert (bool): if true revert order of min_max

    Returns
    -------
        tuple: (min, max)
    """

    # check for invert option
    if invert == True:
        i1, i2 = 1, 0
    else:
        i1, i2 = 0, 1
        
    str_list = str_arg.split('_')
    min, max = float(str_list[i1]), float(str_list[i2])

    return (min, max)



def time_range_parser(time_range_str: str):
    """
    Parses the time range argument to handle the user inputted times.
    
    parameters:
    -----------
    time_range (str): time1_time2 where time1 and 2 can have different
                      time formats.
    """
    
    # check for separator
    try:
        times = time_range_str.split('_')
    except:
        raise ValueError(f'{time_range_str} missing "_" time separator.')

    # check times for non-int
    bInt = check_if_int(times)
    if not bInt:
        raise ValueError(f'Times must be int ({times})')

    # initialize datetime array
    dt_objs = []
    
    for t in times:
        # case YYYYMMDDHH
        if len(t) == 10:
            dt_objs.append(dt_converter(t,"%Y%m%d%H"))

        # case YYYYMMDD
        if len(t) == 8:
            dt_objs.append(dt_converter(t, "%Y%m%d"))

        # case YYYYMM
        if len(t) == 6:
            dt_objs.append(dt_converter(t, "%Y%m"))

        # case YYYY
        if len(t) == 4:
            dt_objs.append(dt_converter(t, "%Y"))
    
    #print(dt_objs)
    return dt_objs

def coefficient_check(args: dict) -> tuple:
    """
    Make sure that all coefficients are entered and none are None
    
    Parameters:
    -----------
    args (dict): argument dictionary
    
    Returns:
    --------
    tuple: coefficient tuple (A, B, C, D, E)
    """

    print('\nChecking for coefficients')
    n_found = 0
    arg_list = ['A', 'B', 'C', 'D', 'E']
    for arg in arg_list:
        if arg in arg_list:
            if args[arg] != None:
                n_found += 1

    if n_found == 0:
        print('No coefficient arguments found\n')
        time.sleep(0.4)
        return None, None, None, None, None
    
    print('{} coefficient arguments found\n'.format(n_found))
    time.sleep(0.4)
    return args['A'], args['B'], args['C'], args['D'], args['E']

    
def check_training(arg_train: bool, arg_retrain: bool, arg_amp_train: bool):
    """
    Checks if training is active based on the inputs.

    Parameters:
    -----------
        arg_train (bool): argument for training
        arg_retrain (bool): argument for retraining
        arg_amp_train (bool): argument for amplitude training
        
    Returns:
    --------
        train_opt (int): integer with selected training option
                        0 - no training
                        1 - retrain / train with amplitude
                        2 - retrain / train without amplitude
                        3 - train without amplitude
    """

    if arg_train and arg_retrain and arg_amp_train:
        train_opt = 1
    elif arg_train and arg_amp_train:
        train_opt = 1
    elif arg_train and arg_retrain:
        train_opt = 2
    elif arg_train:
        train_opt = 3
    else:
        train_opt = 0

    return train_opt