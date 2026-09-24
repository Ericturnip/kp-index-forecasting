"""
MODULE: model_analysis
----------------------
@author: Benjamin Pieczynski
DATE: 2024-05-29

PURPOSE:
    Analysis of model training to verify how effective the model is at predicting
    Kp index within our data-set.
    
INCLUDED FUNCTIONS:
    test_linear_model

MODIFICATION HISTORY:
    2024-06-11 v1.1.0 - testing the linear model allows for model to be 
                        input as a tuple as well.
"""

# imports
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from datetime import timedelta

# user imports
import utils

def test_linear_model(model, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Tests Linear Regression model and returns a statistical analysis of
    said model.

    Parameters:
    -----------
        model (any): trained Linear Regression model from sklearn
        or a tuple of (A, B, C, D, E, *amp)
        X_test (np.ndarray): test data set or test results if rerun mode
        y_test (np.ndarray): expected test results
    
    Returns:
    --------
        mae (float): mean absolute error
        rmse (float): root mean squared error
        correlation (float): correlation coefficient
    """

    # predict and the test set
    if type(model) == tuple:
        try:
            A, B, C, D, E, amp = model
            y_pred = A \
            + (B * X_test['coupling_term'] \
            + C * X_test['viscous_term'] \
            + D * X_test['bz_term'] \
            + E * X_test['dbzdt_term']) \
            * amp

        except:
            A, B, C, D, E = model
            y_pred = A \
                     + B*X_test['coupling_term'] \
                     + C*X_test['viscous_term'] \
                     + D*X_test['bz_term'] \
                     + E*X_test['dbzdt_term']

    else:
        y_pred = model.predict(X_test)
    
    # evaluate the test set
    print('\nEvaluating model...')
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    correlation, _ = pearsonr(y_test, y_pred)

    # extract the coefficients for the linear terms
    if type(model) == tuple:
        pass
    else:
        A = model.intercept_
        B = model.coef_[0]
        C = model.coef_[1]
        D = model.coef_[2]
        E = model.coef_[3]
        amp = 1

    # print values
    print('\nMODEL VALUES:')
    print('--------------')
    time.sleep(0.1)
    print(f'intercept = {A:.10f}')
    time.sleep(0.2)
    print(f'B         = {B:.10f}')
    time.sleep(0.2)
    print(f'C         = {C:.10f}')
    time.sleep(0.2)
    print(f'D         = {D:.10f}')
    time.sleep(0.2)
    print(f'E         = {E:.10f}')
    time.sleep(0.2)
    print(f'amp       = {amp:.10f}')
    time.sleep(0.2)

    print('\nMODEL RESULTS:')
    print('--------------')
    time.sleep(0.1)
    print(f'Mean Absolute Error:     {mae:.6f}')
    time.sleep(0.2)
    print(f'Root Mean Squared Error: {rmse:.6f}')
    time.sleep(0.2)
    print(f'Correlation Coefficient: {correlation:.6f}\n')
    time.sleep(0.2)
    print('COMPLETE\n')
    
    return mae, rmse, correlation
    

def plot_kp_analysis(df: pd.DataFrame, kp_df: pd.DataFrame, out_dir: str, 
                     time_str: str, window_size: float = 1, forecast_time=None,
                     bOld=False):

    print('\nPlotting...')
    
    # Apply boxcar smoothing with a time-based window size
    kp = utils.boxcar_time_smoothing(df, window_size, 'date', 'kp')
    kp_old = utils.boxcar_time_smoothing(df, window_size, 'date', 'old_kp_model')
    kp_new = utils.boxcar_time_smoothing(df, window_size, 'date', 'new_kp_model')
    
    # Create a figure and axis objects
    fig, ax1 = plt.subplots(figsize=(9, 6))
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # boolean filter for forecast
    if forecast_time != None:
        w = kp_df['date'] <= forecast_time + (1 / 24 / 365)
        w2 = kp_df['date'] >= forecast_time + (1 / 24 / 365)
        w2 = np.array(w2, dtype=bool)
        n_future = w2.sum()
    else:
        w = [True for _ in range(len(kp_df['date']))]
        n_future = len(w)

    # Time series plot
    ax1.plot(kp_df['date'][w], kp_df['kp'][w], label='GFZ Kp', color='black', linewidth=2, zorder=2)
    if n_future > 1:
        ax1.plot(kp_df['date'], kp_df['kp'], label='GFZ Kp Future', color='black', linestyle=':', 
                 linewidth=2, zorder=1)
    if bOld:
        ax1.plot(kp_old['date'], kp_old['old_kp_model'], label='Old Model Kp', color='orange',
                 linewidth=2, zorder=9)
    ax1.plot(kp_new['date'], kp_new['new_kp_model'], label='Model Predicted Kp', color='blue', 
             linewidth=2, zorder=10)

    if forecast_time != None:
        #print(forecast_time)
        ax1.axvline(forecast_time, color='red', lw=2, linestyle='--', label='Forecast Time')

    # Formatting the time series plot
    ax1.set_title('Kp-Index Time-Series', fontsize=18, weight='bold', pad=15)
    if forecast_time != None:
        label_str = utils.format_forecast_time(utils.fractional_year_to_datetime(forecast_time))
        ax1.set_xlabel(f'Forecast Date: {label_str}', fontsize=16, weight='bold', labelpad=10)
    else:
        ax1.set_xlabel('Date - UT', fontsize=16, weight='bold', labelpad=10)
    ax1.set_ylabel('Kp-Index', fontsize=16, weight='bold', labelpad=10)
    
    # legend
    legend = ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), borderaxespad=1.5)
    plt.setp(legend.get_texts(), fontproperties=FontProperties(weight='bold', size=12))
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')
    #ax1.grid(True)

    # format the tick labels
    min_t = kp_new['date'].min()
    max_t = kp_new['date'].max()
    start_date = utils.fractional_year_to_datetime(min_t) - timedelta(days=2)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = utils.fractional_year_to_datetime(max_t) + timedelta(days=2)
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    diff = end_date - start_date
    
    # set interval for major tick marks
    max_major_labels = diff.days
    interval = max_major_labels // 6
    
    # major tick marks
    major_ticks = []
    major_labels = []
    n = 0
    current_date = start_date
    while current_date <= end_date:
        cfrac_year = utils.datetime_to_fractional_year(current_date)
        major_ticks.append(cfrac_year)
        if n == 0:
            major_labels.append(utils.dt_axis_format(current_date))
        else:
            major_labels.append('')
        current_date = current_date + timedelta(days=1)
        n+=1
        if n==interval:
            n = 0
    font_properties_major = FontProperties(weight='bold', size=15)
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels(major_labels, 
                        fontproperties=font_properties_major)
    y_major_ticks = np.arange(0, 10.1, 1)
    ax1.set_yticks(y_major_ticks)
    ax1.set_yticklabels(y_major_ticks, FontProperties=font_properties_major)
    ax1.tick_params(axis='x', which='major', width=1.5, length=6.5, direction='in', top=True, pad=7)
    ax1.tick_params(axis='y', which='major', width=1.5, length=6.5, direction='in', right=True, pad=7)
        
    # minor tick marks
    minor_ticks = []
    interval = 12 if diff.days > 20 else 6
    current_date = start_date + timedelta(hours=interval)
    while current_date <= end_date:
        cfrac_year = utils.datetime_to_fractional_year(current_date)
        minor_ticks.append(cfrac_year)
        current_date = current_date + timedelta(hours=interval)
    y_minor_ticks = np.arange(0, 10.1, 0.25)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.set_yticks(y_minor_ticks, minor=True)
    ax1.tick_params(axis='x', which='minor', width=1.5, length=3.5, direction='in', top=True)
    ax1.tick_params(axis='y', which='minor', width=1.5, length=3.5, direction='in', right=True)

    # set splines to be bold
    for spine in ax1.spines.values():
        spine.set_linewidth(2)  # Adjust the width as needed
        spine.set_color('black')  # Set color to black or any desired color
    
    #time_axis = kp_new['date']
    #num_labels = 5
    #ticks = np.linspace(time_axis.min(), time_axis.max(), num_labels)
    #ax1.set_xticks(ticks)
    #ax1.set_xticklabels([utils.frac_date_axis_format(tick) for tick in ticks])

    # set x, y value range
    if forecast_time != None:
        xstart = utils.fractional_year_arithmetic(forecast_time, 1.15, operator='-', units='day')
        xend   = utils.fractional_year_arithmetic(forecast_time, 4, units='day')
    else:
        dates = kp_df['date'].values
        xstart = dates[ 0]
        xend   = dates[-1]
    #print(forecast_time, xstart, xend)
    #xstart, xend = min_t, max_t
    ax1.set_xlim(xstart, xend)
    ax1.set_ylim(-0.2,9.1)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the plot as a PNG file
    filename = os.path.join(out_dir, f'kp_time_series_analysis_{time_str}.png')
    plt.savefig(filename)
    print('COMPLETE:', filename)

    # Scatter plot
    fig, ax2 = plt.subplots(figsize=(8, 6))
    
    if bOld:
        ax2.scatter(kp_old['old_kp_model'], kp['kp'], label='Old Model vs Actual', color='orange', alpha=0.5)
    ax2.scatter(kp_new['new_kp_model'], kp['kp'], label='Predicted Model vs Actual', color='blue', alpha=0.5)

    # Adding a line of best fit for the scatter plot
    if bOld:
        m_old, b_old = np.polyfit(kp_old['old_kp_model'], kp['kp'], 1)
        ax2.plot(kp_old['old_kp_model'], m_old*kp_old['old_kp_model'] + b_old, color='orange')

    m_pred, b_pred = np.polyfit(kp_new['new_kp_model'], kp['kp'], 1)
    ax2.plot(kp_new['new_kp_model'], m_pred*kp_new['new_kp_model'] + b_pred, color='blue')

    # Formatting the scatter plot
    ax2.set_title('Correlation of Kp Index')
    ax2.set_xlabel('Model Kp Index')
    ax2.set_ylabel('Actual Kp Index')
    #ax2.legend()
    ax2.grid(True)
    
    # Compute Pearson correlation coefficient
    if bOld:
        corr_old, _  = pearsonr(kp_old['old_kp_model'], kp['kp'])
        ax2.text(0.05, 0.90, f'Old Model Corr: {corr_old:.3f}', transform=ax2.transAxes, fontsize=12, verticalalignment='top', color='orange')
    corr_pred, _ = pearsonr(kp_new['new_kp_model'], kp['kp'])
    ax2.text(0.05, 0.95, f'Pred Model Corr: {corr_pred:.3f}', transform=ax2.transAxes, fontsize=12, verticalalignment='top', color='blue')

    # set maximums
    ax2.set_ylim(0,8)

    # Set a common title for the entire figure
    #fig.suptitle('Kp Index Analysis', fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the plot as a PNG file
    filename = os.path.join(out_dir, f'kp_correlation_analysis_{time_str}.png')
    plt.savefig(filename)
    print('COMPLETE:', filename)
    
    # plot 2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['date'], df['bz'] / np.median(df['bz']), label='bz')
    ax.plot(df['date'], df['velocity'] / np.median(df['velocity']), label='velocity')
    ax.plot(df['date'], df['density'] / np.median(df['density']), label='density')
    ax.plot(kp_df['date'], kp_df['kp'] / np.median(kp_df['kp']), label='Actual Kp', color='black', linewidth=1.5)

    # tick labels
    num_labels = 6
    ticks = np.linspace(df['date'].min(), df['date'].max(), num_labels)
    ax.set_xticks(ticks)
    ax.set_xticklabels([utils.frac_date_axis_format(tick) for tick in ticks])

    # set axis labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Values')
    ax.legend()

    # Save the plot as a PNG file
    filename = os.path.join(out_dir, f'kp_coefficient_contribution_{time_str}.png')
    plt.savefig(filename)
    print('COMPLETE:', filename)
    
    # Show the plot
    plt.show()
    return