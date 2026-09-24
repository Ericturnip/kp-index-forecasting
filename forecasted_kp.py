#!/home/soft/anaconda3/bin/python3

"""
Created on Sun Jan 9 14:06:24 2022

@author: Matthew Bracamontes

PURPOSE:
    Read e3 and Bxyz_e3 files, to use with kp_from_sw to create a kp index value

INPUTS:
    e3
    Bxyz_e3

OUTPUTS:
    SOURCE.txt

DEPENDENCIES:
    NONE

NOTES:
    Some notes here

    VARIABLES (Important non IO variables):
        NONE

MODIFICATION HISTORY:
    2024-06-27 Benjamin Pieczynski: modified program to work with the
                                    kp-index program packages
"""
from datetime import datetime, timezone
import argparse
#import subprocess
from utils import date_to_doy
import numpy as np
import os
import pandas as pd

# user import
from run_kp_model import kp_model
from read_module import load_model
from data_filter import var_time_shift
from defaults import time_boundaries, cwd
from read_module import read_bxyz, read_e3
from data_acquisition import build_set_dataframe


def format_kp_file(filename, *arrays):
        widths = [5,3,7,7,7,7,7]
        with open(filename, 'w') as f:
            for values in zip(*arrays):
                formatted_values = [f'{val:^{widths[i]}}' for i, val in enumerate(values)]
                line = ' '.join(formatted_values) + '\n'
                f.write(line)
                
def format_noaa_file(filename, *arrays):
        widths = [5,4,7,7]
        with open(filename, 'w') as f:
            for values in zip(*arrays):
                formatted_values = [f'{val:^{widths[i]}}' for i, val in enumerate(values)]
                line = ' '.join(formatted_values) + '\n'
                f.write(line)

def add_header(filename,header):

    # Read the existing content of the file
    with open(filename, 'r') as file:
        content = file.readlines()

        # Add the new row to the beginning
        content.insert(0, header + '\n')

        # Write the new content back to the file
        with open(filename, 'w') as file:
            file.writelines(content)

def split_datetime_strings(datetime_strings):
    years = []
    months = []
    days = []
    hours = []

    for datetime_str in datetime_strings:
        datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
        years.append(datetime_obj.year)
        months.append(datetime_obj.month)
        days.append(datetime_obj.day)
        hours.append(datetime_obj.hour)

    return years, months, days, hours

def read_noaa(noaa_file):

    noaa_array = []
    with open(noaa_file, 'r') as noaa_values:
        for line in noaa_values:
            noaa_array.append(line)
    columns = [line.split(',') for line in noaa_array]
    data_cleaned = [[element.replace('[', '').replace(']', '').replace('"','') for element in row] for row in columns]
    new_2d_array = []
    for column in data_cleaned:
        for i in range(0, len(column), 4):
            new_2d_array.append(column[i:i+4])

    transposed_list = list(map(list, zip(*new_2d_array)))

    # Create new lists from transposed list
    new_lists = []
    for sublist in transposed_list:
        new_list = []
        for value in sublist[1:]:
            new_list.append(value)
        new_lists.append(new_list)
    time = new_lists[0]
    years,months,days,hours = split_datetime_strings(time)
    doy = []
    for year,month,day in zip(years,months,days):
        doys = date_to_doy(year,day,month)
        doy.append(doys)
    kp = new_lists[1]
   
    new_year, new_doy, new_hour, new_kp =[],[],[],[]
    for i in range(len(hours)):
        if hours[i] not in [0, 6, 12, 18]:
            new_year.append(years[i])
            new_doy.append(doy[i])
            new_hour.append(hours[i])
            new_kp.append(kp[i])
    new_hour = [str(num) + '.000' for num in new_hour]
    new_kp = [str(num) + '0' for num in new_kp]
#    print(new_hour)
    return new_year,new_doy,new_hour,new_kp
 
def mimic_e3_format(filename, *arrays):
    widths = [5,3,6,6,6,8,8,8,8,8,8]
    with open(filename, 'w') as f:
        for values in zip(*arrays):
            formatted_values = [f'{val:>{widths[i]}}' for i, val in enumerate(values)]
            line = ' '.join(formatted_values) + '\n'
            f.write(line)
            
def find_first_common(arr1, arr2, arr4,arr5):
    arr1 = list(map(float, arr1))
    arr2 = list(map(float, arr2))
    check_year = is_leap_year(int(arr4[0]))
    for val in arr1:
        if (check_year == False):
            if(val > 365):
                val = val - 365
            else:
                val = val
        else:
            if(val > 366):
                val = val - 366
            else:
                val = val
        if val in arr2:
            return val
    return None

def trim_arrays_at_value(arr1, arr2, value, arr4):
    # Find the index of the first occurrence of the value in each array
    farr1 = list(map(float, arr1))
    farr2 = list(map(float, arr2))
    value = float(value)
    try:
        idx1 = farr1.index(value)
    except:
        check_year = is_leap_year(int(arr4[0]))
        if (check_year == False):
            value_e3 = value + 365
        else:
            value_e3 = value + 366
        idx1 = farr1.index(value_e3)

    idx2 = farr2.index(value)

#    print(idx1,idx2)
    # Trim the arrays from the first occurrence of the value
    trimmed_arr1 = arr1[idx1:]
    trimmed_arr2 = arr2[idx2:]

    return trimmed_arr1, trimmed_arr2, idx1, idx2

def is_leap_year(year):
    if (year % 4 == 0):
        if (year % 100 == 0):
            if (year % 400 == 0):
                return True
            else:
                return False
        else:
            return True
    else:
        return False
###################################### Main ###########################################
def main():
    prog_name = 'forecasted_kp'
    description = '''Program to run Kp index in forecast mode'''
    src_help = '''Path to file'''
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description = description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('e3_file', help = src_help)
    parser.add_argument('bxyz_file', help = src_help)
    parser.add_argument('model_file', default=os.path.join(cwd, 'current.model'), help=src_help)
#    parser.add_argument('SOURCE3', help = src_help)
    args = parser.parse_args().__dict__

    e3_file = args['e3_file']
    bxyz_file = args['bxyz_file']
    model_file = args['model_file']

    file_name = e3_file.split('/')[-1]  # Get the file name
    carrington_rot = file_name.split('_')[1]  # Get the part after the underscore

    # read files into DataFrames
    print('\nReading e3 file...')
    e3_df = read_e3(e3_file, time_boundaries)
    print('Reading bxyz file...')
    bxyz_df = read_bxyz(bxyz_file, time_boundaries)
    print('Bulding DataFrame...')
    df = pd.merge(bxyz_df, e3_df, on=['year', 'doy', 'hour'])
    print(df)
    bz = df['bz']

    # store a place holder for kp values
    kp_place_holder = range(len(df))
    df['kp'] = kp_place_holder

    # load most recent kp model
    print('\nLoading current kp model...')
    #model_file = 'current.model'
    model = load_model(model_file)
    A, B, C, D, E, amp, bshift = model
    print('Coefficients =', model)

    # check for a bshift 
    if bshift != 0:
        print('\nshifting magnetic field')
        shift_cols = ['bx', 'by', 'bz'] # columns to shift
        df = var_time_shift(df, bshift, shift_cols, 
                            replace=True)

    print('\nForecasting Kp')
    kp, _ = kp_model(df, A, B, C, D, E, amp)
#
    num_zeros = 7

    # Create an array of the same size filled with zeros
    noaa_7zeros = [0] * len(df['hour']) * num_zeros
    e3_7zeros = [0] * len(df['hour']) * num_zeros
    e3_ones = [1] * len(df['hour']) * num_zeros

    num_zeros = 9

    # Create an array of the same size filled with zeros
    noaa_9zeros = [0] * len(df['hour']) * num_zeros
    e3_9zeros = [0] * len(df['hour']) * num_zeros

    
    noaa_9zeros = [str(num) + '.000' for num in noaa_9zeros]
    noaa_7zeros = [str(num) + '.000' for num in noaa_7zeros]
    e3_ones = [str(num) + '.000' for num in e3_ones]
    e3_9zeros = [str(num) + '.000' for num in e3_9zeros]
    e3_7zeros = [str(num) + '.000' for num in e3_7zeros]
    
    header = 'Year  DOY   HOUR   DENSITY VELOCITY   BZ     KP'

    year = df['year'][0]
    out_file = f'Planetary_K_index_{year}.txt'
    year_array = [int(year) for year in df['year']]
    hour_array = [int(hour) for hour in df['hour']]
    doy_array  = [int(doy ) for doy  in df['doy' ]]
    format_kp_file(out_file, year_array,doy_array,hour_array,df['density'],
                       df['velocity'],bz,kp)
    
    add_header(out_file,header)

    
    header = ' Year DOY   HOUR                           KP'
    
#    out_file = 'NOAA_K_index.txt'
#    format_kp_file(out_file, noaa_year,noaa_doy,noaa_hour,noaa_kp)
#    add_header(out_file,header)
#
#    out_file = 'NOAA_K_index_e3_format.txt'
#    mimic_e3_format(out_file, noaa_year,noaa_doy,noaa_hour,noaa_7zeros,noaa_7zeros,noaa_9zeros,noaa_kp,noaa_9zeros,noaa_9zeros,noaa_9zeros,noaa_9zeros)
#    add_header(out_file,header)

#    out_file = f'kp_{year}.txt'
    out_file = f'kp_{carrington_rot}'
    print(kp)
#    mimic_e3_format(out_file, df['year'],df['doy'],df['hour'],e3_ones,noaa_7zeros,noaa_9zeros,kp,noaa_9zeros,noaa_9zeros,noaa_9zeros,noaa_9zeros)
    mimic_e3_format(out_file, year_array,doy_array,hour_array,e3_ones,noaa_7zeros,noaa_9zeros,kp,noaa_9zeros,noaa_9zeros,noaa_9zeros,noaa_9zeros)
    add_header(out_file,header)

######################################### Main ####################################################
if (__name__ == '__main__'):
    main()

