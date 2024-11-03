import os
import numpy as np
import pandas as pd
from darts import TimeSeries
from childhope.common import setup_logger
from tqdm import tqdm

logger = setup_logger("childhope.data_preprocessing.helper_functions")

def fill_missing_values(
    df: pd.DataFrame,
    ts_variable: str,
    fill_method: str = 'linear',
    datetime_col: str = 'datetime',
    max_gap: str = '1h'
) -> pd.DataFrame:
    """
    Fill missing values in time series data while respecting maximum allowed gaps.
    
    Strategy:
    1. Identifies gaps larger than max_gap in the time series
    2. Masks values across these large gaps to prevent interpolation over long periods
    3. Applies interpolation only within acceptable gap ranges
    
    Interpolation Methods:
    - 'linear': Fits a linear function between existing points. Best for vital signs 
      that change gradually (e.g., temperature)
    - Other options include 'polynomial', 'spline', which might be more suitable for 
      rapidly changing vitals
    
    Gap Handling:
    - Large gaps (> max_gap) are masked to prevent unreliable interpolation
    - This ensures we don't make assumptions about patient state over long periods
    - For example, with max_gap='1h', we won't interpolate between readings that are 
      more than 1 hour apart
    
    Bidirectional Interpolation:
    - Uses both forward and backward points for more accurate estimation
    - Particularly useful at the edges of gaps where we have data on both sides
    """
    df = df.copy()
    
    # Convert string max_gap (e.g., '1h') to pandas timedelta object
    # This allows for flexible gap specifications (e.g., '30min', '2h', '1d')
    max_gap_td = pd.Timedelta(max_gap)
    
    # Calculate time differences between consecutive readings
    # diff() computes the time elapsed between each row
    time_diff = df[datetime_col].diff()
    gap_mask = time_diff > max_gap_td
    
    # Mask values where gaps are too large to prevent interpolation across them
    # This effectively creates separate segments for interpolation
    df[ts_variable] = df[ts_variable].mask(gap_mask)
    
    # Interpolate missing values using specified method
    # limit_direction='both': Uses points before and after gaps
    # limit_area=None: No restrictions on where interpolation can occur
    df[ts_variable] = df[ts_variable].interpolate(
        method=fill_method,
        limit_direction='both',
        limit_area=None
    )
    
    return df

def prepare_patient_vital_time_series(
    vitals_file_path: str,
    vital_sign_columns: list,
    patient_id_column: str = 'patient_id',
    datetime_column: str = 'datetime',
    fill_method: str = 'linear',
    resample_frequency: str = '60s',
    max_gap: str = '1h',
    min_data_points: int = 10
) -> list:
    """
    Prepare vital signs data into standardized time series objects.
    
    Processing Steps:
    1. Data Loading and Initial Cleaning:
       - Load CSV data into pandas DataFrame
       - Remove rows with null timestamps (unparseable or missing timestamps)
       - Convert string timestamps to pandas datetime objects for time-based operations
       - Set datetime as index for time-based operations
    
    2. Data Type Conversion and Standardization:
       - Convert vital signs to float type for numerical operations
       - Replace sentinel values (-999) with NaN
         (sentinel values often indicate missing or invalid measurements)
    
    3. Time Series Resampling:
       - Resample data to consistent frequency (e.g., '60s' for one reading per minute)
       - Calculate mean values for each time window
       - This handles cases where multiple readings exist in one time window
       - Ensures uniform time steps between measurements
    
    4. Patient-Level Processing:
       - Process each patient's data separately to maintain independence
       - Quality Checks:
         * Skip patients missing entire vital sign columns
         * Skip patients with fewer than min_data_points readings
         (This ensures sufficient data for meaningful analysis)
       - Fill missing values within acceptable gaps using interpolation
         * Different strategies for different vital signs
         * Respects maximum gap constraints
    
    5. Time Series Creation:
       - Convert processed data to Darts TimeSeries objects
       - Darts provides specialized time series functionality:
         * Built-in forecasting capabilities
         * Time series specific transformations
         * Consistent API for different models
       - Track processing statistics for quality control
    
    Args:
        vitals_file_path: Path to CSV file containing vital signs data
        vital_sign_columns: List of column names containing vital measurements
        patient_id_column: Column name containing patient identifiers
        datetime_column: Column name containing timestamps
        fill_method: Interpolation method for missing values
        resample_frequency: Target frequency for regular time series
        max_gap: Maximum allowed gap for interpolation
        min_data_points: Minimum required readings per patient
    
    Returns:
        list: List of Darts TimeSeries objects, one per patient
    """
    logger.info("reading vital dataset from %s", vitals_file_path)
    
    # Step 1: Initial data loading and datetime processing
    vitals_df = pd.read_csv(vitals_file_path, index_col=0)
    logger.info("vital dataset loaded successfully; shape: %s", vitals_df.shape)
    
    vitals_df = vitals_df[vitals_df[datetime_column].notnull()]
    vitals_df[datetime_column] = pd.to_datetime(vitals_df[datetime_column])
    vitals_df = vitals_df.set_index(datetime_column)
    
    # Step 2: Convert data types and standardize values
    vitals_df[vital_sign_columns] = vitals_df[vital_sign_columns].astype(float)
    vitals_df[vital_sign_columns] = vitals_df[vital_sign_columns].replace(-999, np.nan)

    # Step 3: Resample data to consistent frequency
    # Group by patient to maintain separation between different patients' data
    vitals_df = vitals_df.groupby(patient_id_column).resample(resample_frequency)[vital_sign_columns].mean().reset_index()
    logger.info("vital dataset resampled successfully to %s", resample_frequency)

    # Step 4: Process each patient's data
    time_series_list = []
    unique_patients = vitals_df[patient_id_column].unique()
    skip_stats = {'missing_vitals': 0, 'insufficient_data': 0}

    for patient_id in tqdm(unique_patients, desc="Creating time series for patients", total=len(unique_patients)):
        patient_data = vitals_df[vitals_df[patient_id_column] == patient_id].copy()
        
        # Quality checks: Skip patients with invalid or insufficient data
        if patient_data[vital_sign_columns].isna().all().any():
            skip_stats['missing_vitals'] += 1
            continue
            
        if len(patient_data) < min_data_points:
            skip_stats['insufficient_data'] += 1
            continue
        
        # Fill missing values for each vital sign separately
        for column in vital_sign_columns:
            patient_data = fill_missing_values(
                patient_data,
                ts_variable=column,
                fill_method=fill_method,
                datetime_col=datetime_column,
                max_gap=max_gap
            )
        
        # Step 5: Create TimeSeries objects
        patient_data = patient_data.set_index(datetime_column)
        try:
            ts = TimeSeries.from_dataframe(
                patient_data,
                value_cols=vital_sign_columns,
                fill_missing_dates=True,
                freq=resample_frequency
            )
            time_series_list.append(ts)
        except Exception as e:
            logger.warning(f"Failed to create TimeSeries for patient {patient_id}: {str(e)}")
            continue
    
    # Log processing summary
    logger.info("Skipped patients summary:")
    logger.info("- Missing vital signs: %d patients", skip_stats['missing_vitals'])
    logger.info("- Insufficient data points (<%d): %d patients", min_data_points, skip_stats['insufficient_data'])
    logger.info("Time series created successfully for %s patients", len(time_series_list))
    
    return time_series_list
