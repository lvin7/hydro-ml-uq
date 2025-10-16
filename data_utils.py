import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

"""
data_utils.py

This module provides utility functions for loading and preprocessing data for multistep prediction tasks.

Usage:
    -   use data_pred function to prepare data for training, validation, and testing. 
"""

# ----------------------------------
# Helper Functions
# ----------------------------------


def load_exog_data(file_path, nwp, variables, lead_times):
    """Load exogenous data from a CSV file."""
    dfs = {}
    for var in variables:
        for lead_time in lead_times:
            if lead_time not in [24, 48, 72, 96, 120]:
                raise ValueError(f"Lead time {lead_time} is not recognized.")
            df = pd.read_csv(f"{file_path}/{nwp}_{var}_{lead_time}.csv")
            dfs[var][lead_time] = df
    return dfs


def load_target_data(file_path, target):
    """Load target data from a CSV file."""
    if target not in ['Q', 'H']:
        raise ValueError(f"Target {target} is not recognized.")
    df = pd.read_csv(f"{file_path}/{target}.csv")
    return df


def clean_data(df, datetime_col_index):
    """Clean the data by removing duplicates and handling missing values."""
    df = df.drop_duplicates() 
    df = df.dropna() 
    dt = pd.to_datetime(df.iloc[:, datetime_col_index], dayfirst=True)
    df = df[~dt.isna()]  # Remove rows where date conversion failed
    df.index = df.index.tz_localize(None)
    df.index.name = 'datetime'
    return df


def sync_data(exog_dfs, target):
    """Synchronize exogenous and target data on the datetime index."""
    for item in exog_dfs.values():
        common = target.index.intersection(item.index)
        target = target.loc[common]
        item = item.loc[common]
    if common.empty:
        raise ValueError("No common timestamps across target and exogenous data.")
    return exog_dfs, target


def merge_exog(exog_dfs, variables, lead_times, suffix_map=None):
    """Merge multiple exogenous dataframes into a single dataframe."""
    exogs = []
    for h in lead_times:
        parts = []
        for var in variables:
            if suffix_map is None:
                df = exog_dfs[var][h].add_suffix(f'_{var}')
            else:
                df = exog_dfs[var][h].add_suffix(suffix_map.get(var, f'_{var}'))
            parts.append(df)
        merged = pd.concat(parts, axis=1, join='inner')
        exogs.append(merged)
    return exogs


def scale_data(exogs, target, val_start):
    """Scale the data using Min-Max scaling."""
    # Mask for training data
    train_mask = exogs[0].index < val_start
    scaler = MinMaxScaler().fit(exogs[0].loc[train_mask].values)
    # Scale target variable (only for training data)
    exogs_scaled = []
    for x in exogs:
        x_scaled = scaler.transform(x.values)
        exogs_scaled.append(pd.DataFrame(x_scaled, index=x.index, columns=x.columns))
    # Scale the entire dataset using the fitted scalers
    target_scaler = MinMaxScaler().fit(target.loc[train_mask].values)
    y_scaled = target_scaler.transform(target.values)
    endo_scaled = pd.DataFrame(y_scaled, index=target.index)
    return exogs_scaled, endo_scaled, scaler


def prepare_data(data, target_scaled, target, lag, horizon, val_index, test_index, use_q=False, seasonality=False):
    """
    Prepare data for multistep prediction. 
    """
    if horizon < 1:
            raise ValueError("horizon must be >= 1")
    if lag < 0:
            raise ValueError("lag must be >= 0")
    if len(data) < horizon:
            raise ValueError("Provided data has fewer lead DataFrames than the requested horizon")

    # window size for building past target sequence
    ws = lag + horizon

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    n = len(target)
    # iterate over possible starting positions
    for i in range(n - ws - horizon):
            end_ix = i + ws
            # gather input exogenous sequence:
            # The original requested structure: combine lag previous exog values (from lead 1)
            # plus the horizon exog values (one row per lead). We follow the pattern in the prompt.
            try:
                    # part A: lag rows from the lead-1 DataFrame, from i+horizon .. end_ix inclusive (lag rows)
                    seq_x = data[0].iloc[i + horizon : end_ix + 1, :]  # shape (lag, n_features)
            except Exception as e:
                    # if indexing goes out of bounds, stop
                    break

            # this will add (horizon-1) rows (one per future lead)
            for j in range(1, horizon):
                    row_index = end_ix + j
                    if row_index >= len(data[j]):
                            # out of bounds -> stop building further sequences
                            seq_x = None
                            break
                    row = data[j].iloc[[row_index], :].values  # shape (1, n_features)
                    seq_x = np.vstack((seq_x, row))

            # optionally add seasonality feature (if seasonality=True)
            if seasonality:
                    seq_x = np.hstack((seq_x, np.sin(data[0].index.dayofyear.values.reshape(-1,1)/365.25 * 2*np.pi)))
            if seq_x is None:
                    break

            # optionally attach past target (streamflow) values for the lag period (i : end_ix)
            if use_q:
                    q_vals = target_scaled.iloc[i:end_ix].values.reshape(-1, 1)  # shape (lag, 1)
                    # append q array to seq_x
                    seq_x = np.hstack((seq_x, q_vals))

            # output sequence: next `horizon` target values starting from end_ix (end_ix .. end_ix+horizon-1)
            if end_ix + horizon > n:
                    break
            seq_y = target.iloc[end_ix : end_ix + horizon].values

            # assign to split based on end_ix position
            if end_ix >= test_index:
                    X_test.append(seq_x)
                    y_test.append(seq_y)
            elif end_ix >= val_index:
                    X_val.append(seq_x)
                    y_val.append(seq_y)
            else:
                    X_train.append(seq_x)
                    y_train.append(seq_y)

    # convert lists to numpy arrays (object arrays if ragged); prefer to return float arrays if possible
    def to_numpy(arr_list):
            if not arr_list:
                    return np.empty((0,))
            # check if all shapes equal
            shapes = [a.shape for a in arr_list]
            if len(set(shapes)) == 1:
                    return np.stack(arr_list, axis=0)
            # ragged -> return object array
            out = np.empty((len(arr_list),), dtype=object)
            out[:] = arr_list
            return out

    return (
            to_numpy(X_train),
            to_numpy(y_train),
            to_numpy(X_val),
            to_numpy(y_val),
            to_numpy(X_test),
            to_numpy(y_test),
    )


# ----------------------------------- Parameters (to be set as needed) -----------------------------------

# File paths
exog_file_path = '/data/exogenous'   
target_file_path = '/data/target/target.csv'

# Variables and lead times
DEFAULT_HORIZONS = [24, 48, 72, 96, 120]
FEATURE_MAP = {
      'Qp':     {'variables': ['tp_daily'],                         'seasonality': False, 'use_q': True},
      'Qpt':    {'variables': ['tp_daily', 't2m_raw'],              'seasonality': False, 'use_q': True},
      'Qpts':   {'variables': ['tp_daily', 't2m_raw', 'sd_daily'],  'seasonality': False, 'use_q': True},
      'Qptsd':  {'variables': ['tp_daily', 't2m_raw', 'sd_daily'],  'seasonality': True,  'use_q': True},
}
SUFFIX_MAP = {'t2m_raw': '_temp', 'tp_daily': '_precip', 'sd_daily': '_snow'}

# Train/val/test split date
val_start = '2022-07-01'
test_start = '2023-01-01'

# ----------------------------------
# Main data pre-processing function
# ----------------------------------

def data_prep(
    nwp,
    target,
    exog_file_path,
    target_file_path,
    vars='Qpt',
    horizons=DEFAULT_HORIZONS,
    lag=3,
    datetime_col_index=0,
    val_start=val_start,
    test_start=test_start,
    suffix_map=SUFFIX_MAP,
):
    """Main function to load, clean, synchronize, merge, and scale data."""
    variables = FEATURE_MAP[vars]['variables']
    # Load data
    exog_dfs = load_exog_data(exog_file_path, nwp, variables, horizons)
    target_df = load_target_data(target_file_path, target)
    print(f"Loaded target data with shape: {target_df.shape}")
    print(f"Loaded exogenous data for variables: {list(exog_dfs.keys())}")

    # Clean data
    for var in variables:
        for h in horizons:
            exog_dfs[var][h] = clean_data(exog_dfs[var][h], datetime_col_index)
    target_df = clean_data(target_df, datetime_col_index)
    print(f"Cleaned target data with shape: {target_df.shape}")
    print(f"Cleaned exogenous data for variables: {list(exog_dfs.keys())}")

    # Synchronize data
    exog_dfs, target_df = sync_data(exog_dfs, target_df)
    print(f"Synchronized target data with shape: {target_df.shape}")
    print(f"Synchronized exogenous data for variables: {list(exog_dfs.keys())}")

    # Merge exogenous data
    exogs = merge_exog(exog_dfs, variables, horizons, suffix_map)
    print(f"Merged exogenous data: {exogs}")

    # Scale data
    exog_scaled, endo_scaled = scale_data(exogs, target_df, val_start)
    print(f"Scaled exogenous data and target data: {exog_scaled}, {endo_scaled}")

    # Prepare data for modeling
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        exog_scaled,
        endo_scaled,    
        target_df,
        lag,
        len(horizons),
        val_start,
        test_start,
        use_q=FEATURE_MAP[vars]['use_q'],
        seasonality=FEATURE_MAP[vars]['seasonality'],
    )
    print(f"Prepared data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test