"""
MODULE: cli (command line interface)

@author: Benjamin Pieczynski
DATA: 2024-05-23

PURPOSE:
    Build the command line interface for the program.

MODIFICATION HISTORY:
    None
"""

# imports
import argparse

# user imports
from defaults import prog_meta_data, cwd

# creating the parse object
parser = argparse.ArgumentParser(
                    prog=prog_meta_data['program'],
                    description=prog_meta_data['description'],
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# argument help
time_range_help = '''Range of years to build train-testing data for 
                     (format: YYYYMMDD_YYYMMDD, YYYYMM_YYYYMM, or YYYY_YYYY)'''
cme_help        = '''File for specifying CME times to be excluded. (line format:
                     YYYYMMDD_YYYYMMDD or YYYYMMDDHH_YYYYMMDDHH)'''
seasons_help    = '''Observation seasons to include. \n1: March 21st to June 20th,\n 
                     2: June 21st to 22nd September,\n 3: 23rd September to 21st 
                     December,\n 4: 22nd December to March 20th.'''
test_size_help  = '''Size of the test sample relative to all data. Must not be 
                     greater than 0.99. training_data = 1 - test_size'''
e3_dir_help     = '''Path to the directory containing e3 files.'''
kp_dir_help     = '''Path to NOAA or other organization Kp Index values.'''
b_dir_help      = '''Path to bxbybz directories.'''
source_dir_help = '''Path to a source directory which does not require the user
                     to fully type the paths to the e3, kp, and b directories.'''
save_df_help    = '''Stores true to indicate that the DataFrame should be saved
                     as a text file. Saves as YYYY_YYYY_df.txt in the current
                     working directory if -od is not specified.'''
out_dir_help    = '''Path to output directory where all program outputs are
                     saved.'''
reload_help     = '''Reload file to obtain old DataFrame data from a previous
                     session. Use -sd to specify a search path.'''
kp_filter_help  = '''Allows for a minimum and maximum filter with Kp Index. Format
                     is min_max (eg. 2_11)'''
bz_filter_help  = '''Allows for a minimum and maximum filter with Bz values. Format
                     is min_max (eg. 2_11)'''
eda_help        = '''Activates the Exploratory Data Analysis functions.'''
nbins_help      = '''Sets number to bin Kp Data by. Used for EDA mode.'''
othresh_help    = '''Sets the oversampling imbalance threshold for resampling. 
                     Must be given as a float 0.0 < thresh < 1.0.'''
uthresh_help    = '''Sets the undersampling imbalance threshold for resampling. 
                     Must be given as a float 0.0 < thresh < 1.0.'''
ratio_help      = '''Ratio for scaling the sampling up and down for over to under
                     sampling.'''
resample_help   = '''This argument triggers the resampling option in the program.
                     Wether or not a bin is resampled depends on the threshold
                     (-thresh) argument. The ratio of resampling depends on the set
                     proportion(-ratio) argument.'''
bshift_help     = '''Shifts the bField in time by X days.'''
A_help          = '''Input for y_intercept. Use for testing older models and
                     retraining.'''
B_help          = '''Input for coupling coefficient. Use for testing older models and
                     retraining.'''
C_help          = '''Input for viscous coefficient. Use for testing older models and
                     retraining.'''
D_help          = '''Input for the Bz coefficient. D*10^(0.5 - Bz).'''
E_help          = '''Input for the dBz / dt coefficient. D*10^(1 - dBz / dt).'''
test_help       = '''Option for only testing the model.'''
amp_help        = '''Amplitude option for increasing the amplitude of
                     the peaks in the model. This is not a trainable
                     parameter.'''
boxcar_help     = '''Window width for the boxcar smoothing filter in days. Equivalent
                     to total width around a center point.'''
mreload_help    = '''Full path to a .model file to be reloaded into the program.
                     To retrain the model, one must also include -retrain.'''
alpha_help      = '''Parameter for weights on a retrained linear regression. Used to
                     determine approximately how much a coefficient can change.'''
plot_help       = '''Option to plot the time-series.'''
amp_train_help  = '''Activate amplitude training.'''
retrain_help    = '''Activate retraining.'''
forecast_help   = '''Provide a forecast time of the formatted as a carrington rotation time
                     up to 3 decimal places. (ie. 2282.671)'''
no_split_help   = '''Sets model to be trained on all data with no train-test split.'''
model_type_help = '''Select the training backend. "linear" preserves the original
                     coefficient model. "temporal" adds Kp lag-history features and
                     uses a chronological train-test split.'''

# add in the parser arguments
parser.add_argument('-tr', metavar='time_range', required=False,
                    help=time_range_help)
parser.add_argument('-ft', metavar='forecast_time', required=False,
                    default=None, help=forecast_help)
parser.add_argument('-A', metavar='y_intercept', required=False,
                    default=0.05, type=float, help=A_help)
parser.add_argument('-B', metavar='coupling_coefficient', required=False,
                    default=2.44e-4, type=float, help=B_help)
parser.add_argument('-C', metavar='viscous_coefficient', required=False,
                    default=2.844e-6, type=float, help=C_help)
parser.add_argument('-D', metavar='bz_coefficient', required=False,
                    default=0, type=float, help=D_help)
parser.add_argument('-E', metavar='dbzdt_coefficient', required=False,
                    default=0, type=float, help=E_help)
parser.add_argument('-amp', metavar='amplitude', required=False,
                    default=1, type=float, help=amp_help)
parser.add_argument('-at', '--amp_training', required=False,
                    action='store_true', help=amp_train_help)
parser.add_argument('-retrain', '--retrain', action='store_true',
                    required=False, help=retrain_help)
parser.add_argument('-rlm', metavar='model_reload', required=False,
                    default=None, help=mreload_help)
parser.add_argument('-sd', metavar='source_directory', default=None,
                    required=False, help=source_dir_help)
parser.add_argument('-e3d', metavar='e3_directory', default=cwd,
                    required=False, help=e3_dir_help)
parser.add_argument('-kpd', metavar='kp_directory', default=cwd,
                    required=False, help=kp_dir_help)
parser.add_argument('-bd', metavar='b_directory', default=cwd,
                    required=False, help=b_dir_help)
parser.add_argument('-dfr', metavar='reload', default=None,
                    required=False, help=reload_help)
parser.add_argument('-cme', metavar='cme_file',  default=None, 
                    required=False, help=cme_help)
parser.add_argument('-s', metavar='seasons', default='1234', 
                    required=False, help=seasons_help)
parser.add_argument('-ts', metavar='test_size', type=float, 
                    default=0.2, required=False, help=test_size_help)
parser.add_argument('-sdf', '--save_df', action='store_true', 
                    required=False, help=save_df_help)
parser.add_argument('-kpf', metavar='kp_filter', default=None,
                    required=False, help=kp_filter_help)
parser.add_argument('-bzf', metavar='bz_filter', default=None,
                    required=False, help=bz_filter_help)
parser.add_argument('-eda', '--bEDA', action='store_true',
                    required=False, help=eda_help)
parser.add_argument('-nb', metavar='nbins', default=4, type=int,
                    required=False, help=nbins_help)
parser.add_argument('-othresh', metavar='over_resample_threshold', default=0.4,
                    type=float, required=False, help=othresh_help)
parser.add_argument('-uthresh', metavar='under_resample_threshold', default=0.4,
                    type=float, required=False, help=uthresh_help)
parser.add_argument('-rs', '--bResample', action='store_true',
                    required=False, help=resample_help)
parser.add_argument('-ratio', metavar='rebalance_ratio', default=0.2,
                    type=float, help=ratio_help)
parser.add_argument('-bshift', metavar='bshift', type=float,
                    default=0, required=False, help=bshift_help)
parser.add_argument('-test', '--test', action='store_false',
                    help=test_help)
parser.add_argument('-no_split', '--no_split', action='store_true',
                    help=no_split_help)
parser.add_argument('-mt', '--model_type', default='linear',
                    choices=['linear', 'temporal'], required=False,
                    help=model_type_help)
parser.add_argument('-alpha', metavar='alpha', type=float,
                    default=0.4, required=False, help=alpha_help)
parser.add_argument('-plot', '--plot', action='store_true',
                    required=False, help=plot_help)
parser.add_argument('-window', metavar='window_width', type=float,
                    default=1, required=False, help=boxcar_help)
parser.add_argument('-od', metavar='output_directory', default=cwd,
                    required=False, help=out_dir_help)
