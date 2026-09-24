"""
MODULE: model_training
@author: Benjamin Pieczynski
DATE: 2024-05-29

PURPOSE:
    Utilize sklearn to conduct machine learning on the selected data
    for Kp Index.
    
INCLUDED FUNCTIONS:
    model_arrays
    features_df
    linear_fit
    weighted_linear_regression

MODIFICATION HISTORY:
    NONE
"""

# imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares

# user imports
from run_kp_model import kp_model

def features_df(coupling_func: np.ndarray, viscous: np.ndarray, 
                bz_component: np.ndarray, dbzdt_component: np.ndarray) -> pd.DataFrame:
    """
    generates the X DataFrame for training

    parameters
    ----------
    coupling_func (np.ndarray): coupling function array
    viscous (np.ndarray): viscous function array
    bz_component (np.ndarray): new component of Bz data
    dbzdt_component (np.ndarray): new component of dBz / dt data

    returns
    -------
        pd.DataFrame: features data
    """

    df = pd.DataFrame({
        'coupling_term': coupling_func,
        'viscous_term': viscous,
        'bz_term': bz_component,
        'dbzdt_term':dbzdt_component
    })

    return df

def linear_fit(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple:
    """
    linear regression fitting of data

    parameters
    ----------
        X_train (pd.DataFrame): training DataFrame
        y_train (np.ndarray): Kp Index values of training dataset

    returns
    -------
        model: linear regression model fit
    """
    
    # train and fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def set_kp_bounds(lr_coeff: list, alpha: float = 0.4) -> tuple:
    """
    Sets the bounds for the kp values prior to training.

    Parameters:
    -----------
        lr_coeff (list): list of coefficients
        alpha (float, optional): % change of coefficients. Defaults to 0.4.

    Returns:
    --------
        bounds (tuple): containing ([bounds_min], [bounds_max])
    """

    # check alpha parameter
    if alpha <= 0:
        raise ValueError(f'Alpha ({alpha}) must be >= 0')

    # initialize boundary arrays
    bounds_min = []
    bounds_max = []
    
    # iterate through each coefficient
    i = 0
    for coeff in lr_coeff:
        # set boundaries for the y-intercept
        if i == 0:
            bounds_min.append(-np.inf)
            bounds_max.append( np.inf)

        # set boundaries for A, B, C, D, E
        elif i > 0 and i < 6:
            if coeff < 0:
                coeff = 1e-7
            alpha_max = 1e+6 if coeff * (1 + alpha) <= 0 else coeff * (1 + alpha)
            alpha_min = 1e-7 if coeff * (1 - alpha) <= 0 else coeff * (1 - alpha)
            bounds_min.append(alpha_min)
            bounds_max.append(alpha_max)

        # parameters that can be negative
        else:
            bounds_min.append(coeff * (1 - alpha))
            bounds_max.append(coeff * (1 + alpha))           

        i += 1

    bounds = (bounds_min, bounds_max)

    return bounds


def kp_residuals(params: list, X_train: pd.DataFrame, y_train: pd.DataFrame) -> float:
    """
    Get the residuals from the kp linear model with the new parameters. Built 
    for the new version that takes the Bz and dBz / dt coefficients into
    account.
    
    Parameters:
    -----------
    params (list): kp equation parameters in list form
    X_train (pd.DataFrame): X training data
    y_train (pd.DataFrame): y training data
    
    Returns:
    --------
    y_pred - y_test (float): residuals
    
    """
    try:
        A, B, C, D, E, amp = params
    except:
        A, B, C, D, E = params
    y_pred = A \
           + B*X_train['coupling_func'] \
           + C*X_train['viscous'] \
           + D*X_train['bz_component'] \
           + E*X_train['dbzdt_component']
    y_pred * amp
    return y_pred - y_train


def weighted_linear_regression(X_train: pd.DataFrame, y_train: pd.DataFrame, residual_func, 
                               boundary_func, initial_params: list, alpha: float = 0.4) -> list:
    """
    Linear regression designed for retraining with an alpha
    parameter that limits how much the input parameters can
    change.

    Parameters
    ----------
    X_train (pd.DataFrame): X training data
    y_train (pd.DataFrame): y training data
    residual_func (function): residual function to optimize
    boundary_func (function): function to determine boundaries
    intial_params (list): array of initial function parameters
    alpha (float): fractional weights for weighted linear regression
    bounds (tuple): formatted as (min_bounds, max_bounds)
    
    Returns:
    --------
    optimized_parameters (tuple): array of optimized parameters
    """

    bounds = boundary_func(initial_params, alpha)

    # make sure all initial parameters are within the bounds
    guess_params = []
    for p, lower_bound, upper_bound in zip(initial_params, bounds[0], bounds[1]):
        print(p, lower_bound, upper_bound)
        if p < lower_bound:
            p = lower_bound
        elif p > upper_bound:
            p = upper_bound

        # append parameter
        guess_params.append(p)

    # Perform the optimization
    result = least_squares(residual_func, guess_params, bounds=bounds, 
                           args=(X_train, y_train,))

    # Extract the optimized parameters
    optimized_params = result.x
    return tuple(optimized_params)