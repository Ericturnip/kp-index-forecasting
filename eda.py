"""
MODULE: eda (exploratory data analysis)
@author: Benjamin Pieczynski
DATE: 2024-05-29

PURPOSE:
    Perform exploratory data analysis on sample data to get
    the best results for the machine learning program.
    
INCLUDED FUNCTIONS:

MODIFICATION HISTORY:
    NONE
"""

# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def histo_plot(array: np.ndarray, labels: np.ndarray, title: str, 
               out_dir: str, time_range: str) -> None:
    """
    Plots a histogram of data bins

    Parameters:
    -----------
    array (np.ndarray): array of values
    labels (np.ndarray): array of labels
    title (str): title of plot
    out_dir (str): path to output directory
    time_range (str): time range

    Returns:
    --------
    Saves a figure to '/out_dir/histogram_{time_range}.png'
    """
    
    plt.figure(figsize=(10, 8))
    plt.bar(labels, array, width=0.5, edgecolor='black', 
            alpha=0.7, align='center')
    plt.title(title)
    plt.xlabel("Bin Ranges")
    plt.ylabel("Count")
    plt.xticks(labels, rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    filename = os.path.join(out_dir, f'histogram_{time_range}.png')
    print(f'\nsaving histogram... {filename}\n')
    plt.savefig(filename)
    plt.close()
    
    return

def correlation_heatmap(df: pd.DataFrame, out_dir: str, time_range: str) -> None:
    """
    Builds a correlation heatmap plot for a specific data set
    
    Parameters:
    -----------
    df (pd.DataFrame): data
    out_dir (str): path to output directory
    time_range (str): string representing time range

    Returns:
    --------
    saves a figure to '/out_dir/correlation_heatmap_{time_range}.png'
    """

    
    # calculate the correlation matrix
    corr_matrix = df[['bx', 'by', 'bz',
                      'density', 'velocity',
                      'kp']].corr()
    
    print('\nCorrelation Heatmap')
    print(corr_matrix)
    print()

    # create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    
    # save and close
    plt.savefig(os.path.join(out_dir, f'correlation_heatmap_{time_range}.png'))
    plt.close()

    return

def bin_assigner(val, pair_dict: dict):
    """
    Assigns bins based on the pair dictionary

    Args:
        val (any): integer or float value
        pair_dict (dict): dictionary with bin min max pairs (default is <= max)

    Returns:
        key (any): dictionary key
    """
    keys_list = pair_dict.keys()
    for key in pair_dict:
        if val <= pair_dict[key][1] and val > pair_dict[key][0]:
            return key
        
        if key == min(keys_list) and val == pair_dict[key][0]:
            return key
    

def bin_df(title:str, df: pd.DataFrame, col: str, nbins: int = 8,
           othresh: float = 0.4, uthresh: float = 0.4, ratio: float = 0.2,
           bArray: bool = False) -> pd.DataFrame:
    """
    generates a DataFrame for binned values, adds bins to the main DataFrame, and
    suggests oversampling / undersampling.
    
    parameters
    ----------
    title (str): name for DataFrame
    df (pd.DataFrame): main data
    col (str): column to index
    nbins (int): number of bins to make
    othresh (float): imbalance threshold for oversampling
    uthresh (float): imbalance threshold for undersampling
    ratio (float): new samples / total samples in each bin
    bArray (bool): if df input is actually an array
    
    returns
    -------
    df (pd.DataFrame): df with added column
    df_bin (pd.DataFrame): binned df
    """

    # build the histogram bin DataFrame
    if bArray:
        array = df
        df = pd.DataFrame({
            col: array
            })
    else:
        array = df[col]
        
    hist, bin_edges = np.histogram(array, bins=nbins)
    
    # store information in an array
    pair_dict = {}
    bin_numbers = []
    bin_labels  = []
    bin_edge_L  = []
    bin_edge_R  = []
    for n in range(nbins):
        bin_numbers.append(n)
        bin_labels.append('{} - {}'.format(round(bin_edges[n],2), 
                                           round(bin_edges[n+1],2)))
        pair_dict[n] = (bin_edges[n], bin_edges[n+1])
        bin_edge_L.append(bin_edges[n])
        bin_edge_R.append(bin_edges[n+1])
    
    # reconstruct the DataFrame
    df_bin = pd.DataFrame({
        'bin#': bin_numbers,
        'bin_label': bin_labels,
        'l_edge': bin_edge_L,
        'r_edge': bin_edge_R,
        'counts': hist
    })

    # create a bin column in the main DataFrame
    df['bin#'] = df[col].apply(lambda x: bin_assigner(x, pair_dict))

    # evaluate imbalance
    bin_counts = df_bin['counts']
    min_count = bin_counts.min()
    max_count = bin_counts.max()
    resample_options = []
    max_imbalance = []
    min_imbalance = []
    sample_adjustments = []
    
    for _, row in df_bin.iterrows():
        current_count = row['counts']
        option = 'none'
        min_balance = min_count / current_count
        max_balance =  current_count / max_count
        if min_balance <= uthresh:
            option = 'under'
            sample_adjustment = current_count*ratio
        if max_balance <= othresh:
            option = 'over'
            sample_adjustment = current_count*ratio
        if option == 'none':
            sample_adjustment = 0
        
        max_imbalance.append(round(max_balance, 2))
        min_imbalance.append(round(min_balance, 2))
        sample_adjustments.append(int(sample_adjustment))
        resample_options.append(option)


        print(len(sample_adjustments), len(max_imbalance))
    
    df_bin['max%'] = max_imbalance
    df_bin['min%'] = min_imbalance
    df_bin['resample'] = resample_options
    df_bin['adjustment'] = sample_adjustments
        
    print(f'\n{title} - Binned Data, {nbins} bins')
    print(df_bin)

    return df, df_bin
