"""
MODULE : defaults

@author: Benjamin Pieczynski
DATE: 2024-05-23

PURPOSE:
    Store default and global variables for the program.
    
MODIFICATION HISTORY:
    2.0.0 - 2024-07-05: added new global parameters (store original parameters)
"""

# imports
import os
from datetime import datetime, timezone

# Program Metadata
prog_description = "This program acquires data to train the \
                    Kp-Index on IPS Density, Velocity and B \
                    data. We train the equation Kp = C1 + \
                    C2*V^(4/3)*B_t^(2/3)[sin(theta_c / 2)]\
                    ^(8/3) + C3*N^(1/2)*V^2."

prog_meta_data = {
    'program':      'train_kp',
    'author':       'Benjamin Pieczynski',
    'release_date': '2024-05-23',
    'version':      '2.2.0',
    'description':  prog_description
    }

# file formats in the archive
file_formats = {
    'bxyz': 'Bxyz_insitu_e3_YYYY.txt',
    'e3':   'e3_YYYY.txt',
    'kp':   'kp_YYYY.txt'
    }

# datatypes for DataFrame reloading
dtypes = {
    'year':     int,
    'doy':      int,
    'hour':     int,
    'bx':       float,
    'by':       float,
    'bz':       float,
    'density':  float,
    'velocity': float
}

# definition for seasons
season_definitions = {
    '1': {'start': 80,  'end': 171}, # NH Spring
    '2': {'start': 171, 'end': 265}, # NH Summer
    '3': {'start': 266, 'end': 339}, # NH Fall
    '4': {'start': 340, 'end': 79 }  # NH Winter
}

# month dictionary
month_dict = {
    1:  'Jan',
    2:  'Feb',
    3:  'Mar',
    4:  'Apr',
    5:  'May',
    6:  'Jun',
    7:  'Jul',
    8:  'Aug',
    9:  'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec'
}

# global variables
cwd = os.getcwd() # current working directory
min_time = datetime(1990, 1, 1, 1, 0)
max_time = datetime(2099, 12, 30, 23, 59)
time_boundaries = [min_time, max_time] # adjust these in 2100
sw_old_parms = (0.05, 2.244e-4, 2.844e-6, 0, 0, 1)
