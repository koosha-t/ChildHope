import pandas as pd

def fill_missing_values(
    df: pd.DataFrame,
    ts_variable: str,
    fill_method: str = 'ffill',
    datetime_col: str = 'datetime',
    spline_order: int = 2
) -> pd.DataFrame:
    """
    This function fills missing values in a specified time series variable within a DataFrame
    using a selected filling method. Here’s an overview of the available filling methods:

    Filling Methods:
        - 'ffill' (Forward Fill):
            Propagates the last valid observation forward to the next missing value.
            Useful when the most recent value is a reasonable approximation for the missing values.
            Commonly used in time series data where recent observations are relevant for short-term gaps.

        - 'bfill' (Backward Fill):
            Propagates the next valid observation backward to fill in missing values.
            Useful when future values can reasonably approximate previous missing data points.
            Suitable for cases where upcoming information should replace missing values before it.

        - 'linear' (Linear Interpolation):
            Fills missing values by interpolating linearly between adjacent known values.
            Suitable for data that changes gradually over time, allowing a smooth approximation.
            Particularly helpful for continuous variables such as temperature or heart rate.

        - 'spline' (Spline Interpolation):
            Fills missing values using spline interpolation, which fits a smooth curve through the known data points.
            This method is particularly useful for data with curvature or more complex trends.
            The `spline_order` parameter controls the order of the spline (e.g., 2 for quadratic, 3 for cubic).
            Higher-order splines can fit data with more curvature but may overfit in some cases.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the time series data.
        ts_variable (str): The name of the time series variable (column) with missing values to be filled.
        fill_method (str): The filling method to use. Options are:
                           - 'ffill': Forward fill
                           - 'bfill': Backward fill
                           - 'linear': Linear interpolation
                           - 'spline': Spline interpolation (specify order with spline_order)
        datetime_col (str): The name of the datetime column to check and sort by. Default is 'datetime'.
        spline_order (int): The order of the spline for 'spline' method. Default is 2.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled in the specified time series variable.
        
    Usage Example:
        fill_missing_values(df, ts_variable='ECGHR', fill_method='linear', datetime_col='datetime')
    """
    # Ensure the datetime column exists and sort by it
    if datetime_col not in df.columns:
        raise ValueError(f"The specified datetime column '{datetime_col}' does not exist in the DataFrame.")
    
    # Sort by datetime if not already sorted
    df = df.sort_values(by=datetime_col).reset_index(drop=True)

    # Apply the chosen filling method
    if fill_method == 'ffill':
        df[ts_variable] = df[ts_variable].ffill()
    elif fill_method == 'bfill':
        df[ts_variable] = df[ts_variable].bfill()
    elif fill_method == 'linear':
        df[ts_variable] = df[ts_variable].interpolate(method='linear')
    elif fill_method == 'spline':
        df[ts_variable] = df[ts_variable].interpolate(method='spline', order=spline_order)
    else:
        raise ValueError("Invalid fill_method. Choose from 'ffill', 'bfill', 'linear', or 'spline'.")

    return df
