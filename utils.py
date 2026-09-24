"""
MODULE: utils
@authors: Benjamin Pieczynski, Luke Cota
DATE: 2024-05-23

PURPOSE:
    Contains utility functions to assist the main program.
    
INCLUDED FUNCTIONS:
    dt_converter
    days_in_year
    sci_notate
    boxcar_time_smoothing
    df_index_time2yrfrac
    date_to_doy

MODIFICATION HISTORY:
    NONE
"""

# imports
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from sunpy.coordinates.sun import carrington_rotation_time, carrington_rotation_number
except ImportError:
    carrington_rotation_time = None
    carrington_rotation_number = None

# user imports
from defaults import month_dict

def dt_converter(time_str: str, conversion: str):
    """
    Converts a time string to a datetime object
    
    parameters
    ----------
    time_str (str): time to be converted into a datetime object
    conversion (str): tells the converter how to convert the time
                      based off of datetime documentation
    """
    try:
        return datetime.strptime(time_str, conversion)
    except:
        raise ValueError(f'{time_str} unable to convert time')


def days_in_year(year: int) -> int:
    """
    Checks the number of days in the year
    
    parameters
    ----------
    year (int): current year
    
    returns
    -------
    int: number of days in the year
    """
    # Check if the year is a leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 366
    else:
        return 365


def sci_notate(float_val: float) -> str:
    """
    Turns a float into a scientific notation string
    
    Parameters:
    -----------
    float_val (float): value to turn into a scientific notation
    
    Returns:
    --------
    (str): scientific notation
    """
    return f"{float_val:.3e}"


def is_leap_year(year: int) -> bool:
    """
    Check if a given year is a leap year.
    
    Parameters:
    -----------
    year (int): year to check
    
    Returns:
    --------
    (bool): True or False
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def fractional_year_to_datetime(fractional_year: float) -> datetime:
    """
    Converts fractional years to datetime objects.

    Parameters:
    -----------
        fractional_year (float): value for fractional year (e.g. 2022.34556)

    Returns:
    --------
        datetime: datetime object
    """
    
    # split up fractional year and remaining decimal
    year = int(fractional_year)
    remainder = fractional_year - year
    
    # set initial time
    start_of_year = datetime(year, 1, 1)
    
    # determine # of days in the year
    days_in_year = 366 if is_leap_year(year) else 365
    
    # determine time delta object
    days_added = remainder * days_in_year
    dt = timedelta(days=days_added)
    
    return start_of_year + dt


def boxcar_time_smoothing(df: pd.DataFrame, window_width: float, time_col: str, 
                             value_col: str, bFrac=True):
    """
    Time based boxcar smoothing function to smooth over days
    
    Parameters:
    -----------
    df (pd.DataFrame): input data
    window_width (int): window size in days
    time_col (str): name of time column
    value_col (str): name of column where value is stored
    bFrac (bool): Whether to return dates as a fractional year or
                  datetime object. (default is True)

    Returns:
    --------
    pd.DataFrame: boxcar dataframe [time, value]
    """
    
    # if window is 0
    if window_width == 0:
        return df[[time_col, value_col]]
    
    # store smoothing in a dictionary
    boxcar_dict = {
        time_col: [],
        value_col: []
                   }
    
    # get the minimum and maximum times and set widths
    df_min_time, df_max_time = df[time_col].min(), df[time_col].max()
    current_time = df_min_time
    reg_width  = window_width / (365*24)
    lyr_width  = window_width / (366*24)
    reg_radius = reg_width / 2
    lyr_radius = lyr_width / 2

    # build an array for the boxcar centers
    while current_time <= df_max_time:
        # check for leap years
        bLyr = is_leap_year(int(current_time))
        if bLyr:
            width  = lyr_width
            radius = lyr_radius
        else:
            width  = reg_width
            radius = reg_radius

        # grab maximum and minimum times
        min_time = current_time - radius
        max_time = current_time + radius
        w = (df[time_col] <= max_time) & (df[time_col] >= min_time)
        temp_df = df[[time_col, value_col]][w]
        
        # determine if temp_df can be averaged
        if len(temp_df) > 0:
            # take an average
            val_mean = temp_df[value_col].mean()

            # append values to boxcar dictionary
            boxcar_dict[time_col].append(current_time)
            boxcar_dict[value_col].append(val_mean)
        
        # increment time
        current_time = current_time + width
    
    # convert times to datetime objects if required
    if not bFrac:
        datetime_times = []
        for t in boxcar_dict[time_col]:
            datetime_times.append(fractional_year_to_datetime(t))
        boxcar_dict[time_col] = datetime_times
    
    # build df from dictionary
    boxcar_df = pd.DataFrame(boxcar_dict)
    
    return boxcar_df


def fractional_year(year: int, doy: int, hour: int) -> float:
    """
    Provides a fractional year from a year / doy / hour input.

    Parameters:
    -----------
        year (int): target year
        doy (int): target day of year
        hour (int): target hour

    Returns:
    --------
        fractional_year (float): fractional year
    """
    bLyr = is_leap_year(year)
    yearly_hours = 366*24 if bLyr else 365*24
    fractional_year = (hour + doy*24) / yearly_hours + year
    return fractional_year


def df_index_time2yrfrac(df: pd.DataFrame, index_array: np.ndarray) -> list:
    """
    Function to convert a time in Year, doy, hour format into
    a year fraction

    Parameters:
    -----------
        df (pd.DataFrame): data with values stored in columns
        index_array (np.ndarray): array of indices to select

    Returns:
    ---------
        date_array (list): converted dates
    """
    date_array = []
    for i in index_array:
        row = df.loc[i]
        frac_yr = fractional_year(int(row['year']), int(row['doy']), int(row['hour']))
        current_time = round(frac_yr, 6)
        date_array.append(current_time)
        
    return date_array


def date_to_doy(year, dom, month):
    """
    Function by Luke Cota
    """
    jand, febd, febdl, mard, aprd, mayd = 31, 28, 29, 31, 30, 31
    jund, juld, augd, sepd, octd, novd = 30, 31, 31, 30, 31, 30
    b_lyr = is_leap_year(year)
    
    if (month == 1):
        return dom
    elif (month == 2):
        return jand + dom
    elif (month == 3):
        if b_lyr:
            return jand + febdl + dom
        else:
            return jand + febd + dom
    elif (month == 4):
        if b_lyr:
            return jand + febdl + mard + dom
        else:
            return jand + febd + mard + dom
    elif (month == 5):
        if b_lyr:
            return jand + febdl + mard + aprd + dom
        else:
            return jand + febd + mard + aprd + dom
    elif (month == 6):
        if b_lyr:
            return jand + febdl + mard + aprd + mayd + dom
        else:
            return jand + febd + mard + aprd + mayd + dom
    elif (month == 7):
        if b_lyr:
            return jand + febdl + mard + aprd + mayd + jund + dom
        else:
            return jand + febd + mard + aprd + mayd + jund + dom
    elif (month == 8):
        if b_lyr:
            return jand + febdl + mard + aprd + mayd + jund + juld + dom
        else:
            return jand + febd + mard + aprd + mayd + jund + juld + dom
    elif (month == 9):
        if b_lyr:
            return jand + febdl + mard + aprd + mayd + jund + juld + augd + dom
        else:
            return jand + febd + mard + aprd + mayd + jund + juld + augd + dom
    elif (month == 10):
        if b_lyr:
            return jand + febdl + mard + aprd + mayd + jund + juld + augd + sepd + dom
        else:
            return jand + febd + mard + aprd + mayd + jund + juld + augd + sepd + dom
    elif (month == 11):
        if b_lyr:
            return jand + febdl + mard + aprd + mayd + jund + juld + augd + sepd + octd + dom
        else:
            return jand + febd + mard + aprd + mayd + jund + juld + augd + sepd + octd + dom
    elif (month == 12):
        if b_lyr:
            return jand + febdl + mard + aprd + mayd + jund + juld + augd + sepd + octd + novd + dom
        else:
            return jand + febd + mard + aprd + mayd + jund + juld + augd + sepd + octd + novd + dom
    else:
        return -1

def cr_to_date(cr_number: float) -> datetime:
    """
    grab the date from the carrington rotation number
    
    Parameters:
    -----------
        cr_number (float): Carrington rotation number
        
    Returns:
    --------
        date (datetime)
    """
    
    if carrington_rotation_time is None:
        raise ImportError("sunpy is required for Carrington rotation conversions")

    # Calculate the date
    date = carrington_rotation_time(cr_number).datetime
    
    return date


def date_to_cr(date: datetime, bRound: bool = False):
    """
    Use a datetime object get the Carrington rotation number
    
    Parameters:
    -----------
        date (datetime): date to get CR number
        bRound (bool): optional True / False
    
    Returns:
    --------
    cr_number (float): Carrington rotation number
    """
    
    if carrington_rotation_number is None:
        raise ImportError("sunpy is required for Carrington rotation conversions")

    # use sunpy to get cr number
    cr_number = carrington_rotation_number(date)
    
    # round to 4 decimal places
    if bRound:
        return round(cr_number, 4)

    return cr_number


# Convert fractional years to datetime
def fractional_year_to_datetime(fractional_year: float):
    year = int(fractional_year)
    remainder = fractional_year - year
    start_of_year = pd.Timestamp(year=year, month=1, day=1)
    next_year = pd.Timestamp(year=year+1, month=1, day=1)
    return start_of_year + (next_year - start_of_year) * remainder

    
def datetime_to_fractional_year(dt):
    """
    convert a datetime to fractional years
    """
    year = dt.year
    start_of_year = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    days_in_year = (next_year - start_of_year).days
    elapsed_days = (dt - start_of_year).days \
                   + (dt - start_of_year).seconds \
                   / (24 * 3600)
    fractional_year = year + elapsed_days / days_in_year
    return fractional_year

    
def fractional_year_scaling(units: str, bLyr: bool = False) -> float:
    """
    Computes the fractional scaling for units in a fractional year
    for leap and non-leap years.

    Parameters:
    -----------
        units (str): 'yr', 'day', 'hr', 'min', 'sec'
        bLyr (bool): T / F if it is a leap year

    Returns:
    --------
        scale (float): year scaling factor for the unit
    """

    if units == 'sec':
        scale_ly  = 1 / (60*24*356)
        scale_reg = 1 / (60*24*355)

    elif units == 'min':
        scale_ly  = 1 / (60*24*356)
        scale_reg = 1 / (60*24*355)
    
    elif units == 'hr':
        scale_ly  = 1 / (24*356)
        scale_reg = 1 / (24*355)
    
    elif units == 'd':
        scale_ly  = 1 / 356
        scale_reg = 1 / 355
    
    elif units == 'yr':
        scale_ly  = 1
        scale_reg = 1

    else:
        raise ValueError(f'{units} not in accepted units' \
                         ' ("sec", "min", "hr", "d", "yr")')
        
    # grab scale
    scale = scale_ly if bLyr else scale_reg

    return scale


def fractional_year_arithmetic(frac_year: float, dt: float, operator: str='+', units='hr') -> float:
    """
    Performs fractional year arithmetic (+/-). Takes into account leap
    years.

    Args:
        frac_year (float): current frac_year
        dt (float): value for time difference
        units (str, optional): unit scale (sec, min, hr, d, yr). Defaults to 'hr'.

    Returns:
        float: the calculated time
    """
    # check operator
    if operator not in ['+', '-']:
        raise ValueError(f"Operator ({operator}) no in ['+', '-']")
    
    # conver to datetime
    t0 = fractional_year_to_datetime(frac_year)
    
    # perform timedelta operation
    if units == 'sec':
        if operator == '-':
            t1 = t0 - timedelta(seconds=dt)
        else:
            t1 = t0 + timedelta(seconds=dt)
    elif units == 'min':
        if operator == '-':
            t1 = t0 - timedelta(minutes=dt)
        else:
            t1 = t0 + timedelta(minutes=dt)
    elif units == 'hr':
        if operator == '-':
            t1 = t0 - timedelta(hours=dt)
        else:
            t1 = t0 + timedelta(hours=dt)
    elif units == 'day':
        if operator == '-':
            t1 = t0 - timedelta(days=dt)
        else:
            t1 = t0 + timedelta(days=dt)
    elif units == 'yr':
        if operator == '-':
            t1 = t0 - timedelta(years=dt)
        else:
            t1 = t0 + timedelta(years=dt)
    else:
        raise ValueError(units, "not in ['sec', 'min', 'hr', 'day', 'yr']")

    # convert back into fractional year
    frac_year = datetime_to_fractional_year(t1)
    
    # get max scaling
    #max_scale = fractional_year_scaling(units)

    # determine the new fractional year
    #current_time = frac_year
    #current_year = int(frac_year)
    #dt_remaining = abs(dt) # store remaining delta
    #end_year = int(current_time + dt*max_scale)
    #step = -1 if end_year < int(frac_year) else 1
    
    ## loop through all potential years
    #for yr in range(current_year, end_year + step, step):
    #    
    #    # make sure there is available delta remaining
    #    if dt_remaining >= 0:
    #        bLyr = is_leap_year(yr)
    #        scale = fractional_year_scaling(units, bLyr=bLyr)
    #        
    #        # determine the remaining delta
    #        year_remaining = abs(current_time - yr)
    #        scaled_remaining = year_remaining / scale
    
    #        # determine arithmetic operation needed
    #        if scaled_remaining >= dt_remaining:
    #            current_time += dt_remaining*step*scale
    #            dt_remaining = 0
    #            
    #        else:
    #            current_time += year_remaining*step
    #            dt_remaining -= scaled_remaining

    return frac_year

    


def df_derivative(df, val_col, hour_step: int = 12,
                  max_step_back: int = 10, 
                  tol: float = 1 / (366*24)):
    
    # array to store derivatives
    dbzdt = []
    
    # iterate backwards through the DataFrame
    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i]
        c_bz = row['bz']
        c_time = fractional_year(int(row['year']), 
                                 int(row['doy']), 
                                 int(row['hour']))
        targ_time = fractional_year_arithmetic(c_time, -hour_step) # units in hours
        targ_high = targ_time + tol
        targ_low  = targ_time - tol

        # iterate backwards from time to locate previous time.
        stop = i - (max_step_back + 1)
        for j in range(i - 1, stop, 1):
            try:
                if j < 0:
                    dbzdt.append(np.nan)
                s_row = df.iloc[j]
                s_time = fractional_year(int(s_row['year']), 
                                         int(s_row['doy']), 
                                         int(s_row['hour']))
                if s_time < targ_high and s_time > targ_low:
                    dbzdt.append((c_bz - s_row['bz']) / (hour_step / 24)) # bz / day
            except:
                dbzdt.append(np.nan)
                break

    # invert and transform the array into a numpy array
    dbzdt = np.array(dbzdt)[::-1]

    return df[val_col].to_numpy()

def doy_to_month_day_hour(year: int, doy: float) -> tuple:
    """
    Takes a day of year and converts it into a month

    Parameters:
    -----------
        year (int): target year
        doy (float / int): day of year
        
    Returns:
    --------
        tuple: (month: int, day:int, hour:int)
    """
    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return (date.month, date.day, date.hour)


def frac_date_axis_format(frac_year: float):
    """
    Formats fractional year to a date for the axis of a plot.

    Parameters:
    -----------
        frac_year (float): fractional year
    
    Returns:
    --------
        str: date string in axis format
    """
    year = int(frac_year)
    bLyr = is_leap_year(frac_year)
    days_in_year = 356 if bLyr else 355
    doy = (frac_year - year) * days_in_year
    month, day, hour = doy_to_month_day_hour(year, doy)
    month = month_dict[month]
    formatted_year = year % 100
    return f'{formatted_year:02}-{month}-{day:02}T{hour:02}'

def dt_axis_format(dt: datetime) -> str:
    return '{}/{}'.format(dt.month, dt.day)

def format_forecast_time(dt: datetime) -> str:
    return '{}-{}-{}T{:02} UT'.format(dt.year, dt.month, dt.day, dt.hour)
