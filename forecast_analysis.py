"""
MODULE: forecast_analysis
@author: Benjamin Pieczynski
DATE: 2024/08/02

PURPOSE:
    Provide plotting and statistical analysis of the Kp forecast
    predictions.
    
INCLUDES:
    stat_forecast
    plot_forecast

MODIFICATION HISTORY:
    None
"""

# imports
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from datetime import timedelta

# user imports
import utils

def plot_forecast(df: pd.DataFrame, out_dir: str, kp_df: pd.DataFrame, 
                  window: float, time_str: str):
    print(kp_df)
    print('\nPlotting Comparison...')
    
    # Create a figure and axis objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Time series plot
    ax1.plot(kp_df['date'], kp_df['kp'], label='GFZ Kp', color='black', linewidth=2)
    ax1.plot(df['date'], df['kp_pred'], label='Forecasted Kp', color='maroon', linestyle='--', linewidth=2)


    # Formatting the time series plot
    ax1.set_title('Kp Index Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Kp Index')
    ax1.legend()
    ax1.grid(True)

    # format the tick labels
    time_axis = df['date']
    num_labels = 6
    ticks = np.linspace(time_axis.min(), time_axis.max(), num_labels)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([utils.frac_date_axis_format(tick) for tick in ticks])

    # Scatter plot
    ax2.scatter(df['kp_pred'], df['kp'], label='Forecast vs Actual', color='maroon', alpha=0.5)

    # Adding a line of best fit for the scatter plot
    m_new, b_new = np.polyfit(df['kp_pred'], df['kp'], 1)
    ax2.plot(df['kp_pred'], m_new*df['kp_pred'] + b_new, color='black')


    # Formatting the scatter plot
    ax2.set_title('Kp Forecast Correlation')
    ax2.set_xlabel('Forecasted Kp Index')
    ax2.set_ylabel('GFZ Kp Index')
    #ax2.legend()
    ax2.grid(True)
    
    # Compute Pearson correlation coefficient
    corr_pred, _ = pearsonr(df['kp_pred'], df['kp'])
    ax2.text(0.05, 0.90, f'Pred Model Corr: {corr_pred:.3f}', transform=ax2.transAxes, 
             fontsize=12, verticalalignment='top', color='black')

    # Set a common title for the entire figure
    fig.suptitle(f'Kp Forecast Analysis, Window={window} H', fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the plot as a PNG file
    filename = os.path.join(out_dir, f'kp_forecast_analysis_{time_str}.png')
    plt.savefig(filename)
    print('COMPLETE:', filename)
    
    # Show the plot
    plt.show()
    return