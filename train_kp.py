#!/home/soft/anaconda3/bin/python3

"""
PROGRAM: train_kp

@author: Benjamin Pieczynski
DATE: 2024-05-23

PURPOSE:
    To acquire data, filter it and determine the best weights and 
    coefficients for the Kp Index equation.
    
MODIFICATION HISTORY:
    1.0.0 - 2024-05-23: initial release
    1.1.0 - 2024-06-10: added b-field time-shift and other test options
    1.2.0 - 2024-06-13: added plotting options and boxcar algorithm for smoothing
    1.3.0 - 2024-06-20: added model reloading and saving. Fixed resampling
    1.4.0 - 2024-06-24: major update to model reloading and saving. Program can
                        now plot changes between different models. Fine tuning
                        for most of the functions and a rework of the boxcar 
                        function. Model retraining now working.
    2.0.0 - 2024-07-05: major updates to how the program functions. The program
                        now considers 5 initial parameters and now has an option
                        for training amplitude. The original sw model is now used as
                        the starting parameters if no other options are provided. One
                        can operate the program in test mode using -test.
    2.0.1 - 2024-07-10: some minor updates to fix bugs with the 5 initial parameter
                        option.
    2.0.2 - 2024-07-11: Added in a forecast mode option.
    2.0.3 - 2024-07-22: Added a bz filter
    2.1.0 - 2024-07-29: 4th parameter now determined using dip finder. Some promising
                        results, not sure if it will work long term.
    2.2.0 - 2024-08-02: New plotting system, added forecast program that can loop through
                        different forecasts. Forecast time now determined 5 days prior,
                        some bug fixing.
    2.2.1 - 2024-08-21: Bug fixing with plotting issues.
"""

# imports
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import timedelta

# user imports
from arg_handler import time_range_parser, arg_min_max, coefficient_check, check_training
from cli import parser
from defaults import prog_meta_data, sw_old_parms
from data_acquisition import acquire_data, acquire_reload, forecast_load, read_kp
from data_filter import hi_low_filter, var_time_shift
from eda import correlation_heatmap, bin_df, histo_plot
from model_training import features_df, linear_fit, weighted_linear_regression
from model_training import kp_residuals, set_kp_bounds
from read_module import load_model
from run_kp_model import run_linear_kp_model, kp_model_arrays
from sampler import resampler
from write import write_df, write_linear_model
import utils

# program
def train_kp():
    # prog statement
    print('\n-----------------------------------------------------------')
    print(f"PROGRAM: {prog_meta_data['program']} v{prog_meta_data['version']}")
    print(f"BY: {prog_meta_data['author']}")
    print(f"RELEASED: {prog_meta_data['release_date']}")
    print('-----------------------------------------------------------')
    
    # get the user provided parser arguments
    print('\nparsing arguments...')
    args = parser.parse_args().__dict__
    tr_str        = args['tr'  ]

    # check for forecasts and set plotting parameters
    if args['ft'] != None:
        bForecast = True
        forecast_time = float(args['ft'])
        plot_time_str = str(forecast_time)
    else:
        bForecast = False
        forecast_time = None
        plot_time_str = tr_str
    
    time_range = None if bForecast == True else time_range_parser(args['tr'])
    source_dir    = args['sd'  ]
    e3_dir        = args['e3d' ]
    kp_dir        = args['kpd' ]
    bxyz_dir      = args['bd'  ]
    df_path       = args['dfr' ]
    cme_file      = args['cme' ]
    seasons       = args['s'   ]
    test_size     = args['ts'  ]
    bEDA          = args['bEDA']
    nbins         = args['nb'  ]
    othresh       = args['othresh'  ]
    uthresh       = args['uthresh'  ]
    bResample     = args['bResample']
    bshift        = args['bshift'   ]
    out_dir       = args['od']
    bSaveDF       = args['save_df']
    alpha         = args['alpha']
    A, B, C, D, E = coefficient_check(args) # double checking
    amp           = args['amp']
    window_size   = args['window']
    model_file    = args['rlm']
    bAmp          = args['amp_training']
    bPlot         = args['plot']
    bNosplit      = args['no_split']
    model_type    = args['model_type']

    # check for kp_filters
    if args['kpf'] != None:
        kp_filter = arg_min_max(args['kpf'])
    else:
        kp_filter = None

    # check for bz_filters
    if args['bzf'] != None:
        bz_filter = arg_min_max(args['bzf'])
    else:
        bz_filter = None
    
    
    # set training options
    train_opt = check_training(args['test'], args['retrain'], bAmp)
    bTrain = True if train_opt != 0 else False
    
    print('COMPLETE\n')
    time.sleep(0.5)
    
    # load in a model if needed
    if model_type != 'temporal' and model_file != None:
        A, B, C, D, E, amp, bshift = load_model(model_file)
    elif model_type != 'temporal' and bTrain and None in [A, B, C, D, E, bshift]:
         raise ValueError('Previous parameters must be provided for coefficients' / 
                          'when retraining.')
    else:
        pass

    # if there is a source directory provided change each directory
    if source_dir != None:
        e3_dir     = os.path.join(source_dir, e3_dir)
        bxyz_dir   = os.path.join(source_dir, bxyz_dir)
        kp_dir     = os.path.join(source_dir, kp_dir)
        if df_path is not None:
            df_path  = os.path.join(source_dir, df_path)

    # DataFrame outfile
    df_outfile = os.path.join(out_dir, f'{tr_str}_df.txt')
            
    # grab data from archives or previous files
    if df_path is not None:
        df = acquire_reload(df_path, time_range, seasons)
        kp_df = df
    elif bForecast:
        print('\nACTIVATING FORECAST MODE\n')
        df, kp_df = forecast_load(args['ft'], bxyz_dir, e3_dir, 
                                  kp_dir)
        times_frac = utils.df_index_time2yrfrac(df, df.index)
        max_frac = max(times_frac)
        forecast_time = utils.fractional_year_to_datetime(max_frac) - timedelta(days=5)
        forecast_time = utils.datetime_to_fractional_year(forecast_time)
        plot_time_str = args['ft']
        bTrain = False
        #for _, row in df.iterrows():
        #    print(row['year'], row['doy'], row['hour'], row['kp'])
    else:
        df, kp_df = acquire_data(bxyz_dir, e3_dir, kp_dir,
                          time_range, seasons)

    # set plotting parameters
    plotting_params = (
        plot_time_str,     # for file name
        bPlot,        # whether or not to plot 
        window_size,  # boxcar function
        forecast_time # None or actual time
        )

    # shift be values if applicable
    if bshift != 0:
        print(f'\nShifting B-field in time: {bshift} days...')
        shift_cols = ['bx', 'by', 'bz'] # columns to shift
        df = var_time_shift(df, bshift, shift_cols)
        print('shift COMPLETE\n')
        time.sleep(0.5)
    
    # filter out Kp values
    if kp_filter != None:
        print('\nFiltering kp values: min - {} max - {}'.format(kp_filter[0], 
                                                                kp_filter[1]))
        df = hi_low_filter(df, kp_filter[1], kp_filter[0], 'kp')
        print('Filtering COMPLETE\n')
        time.sleep(0.2)

    # filter out Bz values
    if bz_filter != None:
        print('\nFiltering bz values: min - {} max - {}'.format(bz_filter[0], 
                                                                bz_filter[1]))
        df = hi_low_filter(df, bz_filter[1], bz_filter[0], 'bz')
        print('Filtering COMPLETE\n')
        time.sleep(0.2)

    if model_type == 'temporal':
        from temporal_model import (
            evaluate_saved_temporal_model,
            load_temporal_model,
            train_temporal_model,
        )

        if bSaveDF:
            print('\nSaving DataFrame...')
            time.sleep(0.2)
            write_df(df, df_outfile)
            print('SUCCESS...', df_outfile, '\n')
            time.sleep(0.2)

        if bNosplit:
            raise ValueError('Temporal model requires a chronological test split. Remove -no_split.')

        if not bTrain:
            if model_file is None:
                raise ValueError('Temporal test mode requires -rlm <temporal_model.pkl>.')

            print('\nEvaluating saved temporal model...')
            temporal_bundle = load_temporal_model(model_file)
            metrics, prediction_df = evaluate_saved_temporal_model(temporal_bundle, df)
            print('\nTEMPORAL MODEL RESULTS:')
            print('-----------------------')
            print(f"Mean Absolute Error:     {metrics['mae']:.6f}")
            print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
            print(f"Correlation Coefficient: {metrics['corr']:.6f}")
            return

        print('\n### Temporal Model Comparison ###')
        print('--------------------------------\n')
        time.sleep(0.6)

        model_name = args['tr'] if args['tr'] is not None else 'temporal_model'
        dataset_path = df_path if df_path is not None else 'acquired_from_archives'
        temporal_results = train_temporal_model(
            df,
            test_size=test_size,
            alpha=alpha,
            output_dir=out_dir,
            model_name=model_name,
            dataset_path=dataset_path,
        )

        nowcast_df = temporal_results['nowcast_df']
        solar_df = temporal_results['solar_df']
        recursive_df = temporal_results['recursive_df']

        print('BEST NOWCAST MODELS:')
        print(nowcast_df[['model', 'mae', 'rmse', 'corr']].head(6).to_string(index=False))
        print()
        print('BEST SOLAR-WIND-ONLY MODELS:')
        print(solar_df[['model', 'mae', 'rmse', 'corr']].head(5).to_string(index=False))
        print()
        print('RECURSIVE FORECAST SNAPSHOT:')
        print(recursive_df[['model', 'horizon', 'mae', 'rmse', 'corr']].head(12).to_string(index=False))
        print()
        print('Temporal comparison outputs saved:')
        print(f"  model:   {temporal_results['model_path']}")
        print(f"  summary: {temporal_results['summary_path']}")
        print(f"  report:  {temporal_results['report_path']}\n")

        print('\nPROGRAM COMPLETE')
        return

    # write DataFrame for reloading
    if bSaveDF:
        print('\nSaving DataFrame...')
        time.sleep(0.2)
        write_df(df, df_outfile)
        print('SUCCESS...', df_outfile, '\n')
        time.sleep(0.2)

    # section for exploratory data analysis
    if bEDA:
        # create a correlation heatmap, histogram, and bin the data
        correlation_heatmap(df, out_dir, tr_str)
        time.sleep(2)
        df, binned_df = bin_df('Kp Index', df, 'kp', 
                               nbins=nbins, othresh=othresh, uthresh=uthresh)
        time.sleep(2)
        histo_plot(binned_df['counts'], binned_df['bin_label'], 'Kp Index',
                   out_dir, tr_str)
        
    ### machine learning ###

    # get base model (X), prediction targets (y)
    print('gathering model data...')
    coupling_func, viscous, bz_term, dbzdt_term = kp_model_arrays(df)
    X = features_df(coupling_func, viscous, bz_term, dbzdt_term)
    y = df['kp']
    print('X', len(X), 'y', len(y))
    print('COMPLETE\n')
    time.sleep(0.2)

    # option to run a test on new data with previous coefficients
    if bTrain == False:
        params = A, B, C, D, E, amp
        run_linear_kp_model(params, sw_old_parms, 
                            X, y, df,
                            out_dir, kp_df, plotting_params=plotting_params)
        return

    print('\n### Starting Machine Learning ###')
    print('---------------------------------\n')
    time.sleep(1)

    # build train test split and bin data if resampling
    print('building train test split...')
    if bNosplit:
        X_train, X_test, y_train, y_test = (X, X, y, y)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=42)
    
    print('COMPLETE\n')
    time.sleep(0.2)

    if bResample:
        print('\nresampling training set...')
        
        # reset the indices for each X, y training set
        X_train = X_train.reset_index().drop('index', axis=1)
        y_train = pd.DataFrame({'kp': y_train.values})
        
        # bin the kp index to determine over or under samplint
        y_train, binned_df = bin_df('Resampling Bins', y_train, 'kp', 
                              nbins=nbins, othresh=othresh, uthresh=uthresh)
        print()
        
        # resample data and only store values from kp
        X_train, y_train = resampler(X_train, y_train, binned_df)
        y_train = y_train['kp'].values
        print('Resampling - COMPLETE')
        time.sleep(0.2)

    ## scipy least-squares regression - train amplitudes##
    if train_opt in [1, 2]:
        print('\n### Scipy Least-Squares Regression Model ###')
        print('--------------------------------------------\n')
        time.sleep(0.6)
        
        # store old parameters
        old_params = (A, B, C, D, E, amp)

        # determine parameters to pass into LR
        if bAmp:
            lr_coeff = [A, B, C, D, E, amp]
        else:
            lr_coeff = [A, B, C, D, E]
            
        print('fitting model...')
        model = weighted_linear_regression(X_train, y_train, kp_residuals,
                                           set_kp_bounds, lr_coeff, alpha=alpha)
        print('COMPLETE\n')
        time.sleep(0.2)

    elif train_opt == 3:
        ## linear regression - no amplitude training ##
        print('\n### Linear Regression Model ###')
        print('-------------------------------\n')
        time.sleep(0.6)

        # fit model
        print('fitting model...')
        model = linear_fit(X_train, y_train)
        time.sleep(0.2)
        print('COMPLETE\n')
        time.sleep(0.2)

        old_params = sw_old_parms
        
        # get coefficients
        A = model.intercept_
        B = model.coef_[0]
        C = model.coef_[1]
        D = model.coef_[2]
        E = model.coef_[3]

        # save model as tuple
        model = (A, B, C, D, E, amp)
    
    # plot model
    mae, rmse, corr = run_linear_kp_model(model, old_params,
                                          X_test, y_test, df,
                                          out_dir, kp_df, plotting_params=plotting_params)

    # write the model
    write_linear_model(out_dir, args['tr'], model,
                       mae, rmse, corr,
                       bshift=bshift)
    
    print('\nPROGRAM COMPLETE')
    
    return

#########################################################################

# program call
if __name__ == '__main__':
    train_kp()
