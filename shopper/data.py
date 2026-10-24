from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'online_shoppers_intention.csv'
NUMERIC = ['Administrative', 'Administrative_Duration', 'Informational',
           'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
           'BounceRates', 'ExitRates', 'SpecialDay']
CATEGORICAL = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
               'VisitorType', 'Weekend']
FEATURES = NUMERIC + CATEGORICAL
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def load_data(path=DATA):
    df = pd.read_csv(path)
    expected = set(FEATURES + ['PageValues', 'Revenue'])
    if set(df.columns) != expected:
        raise ValueError(f'Unexpected schema: missing={expected-set(df.columns)}, extra={set(df.columns)-expected}')
    for name in ['Revenue', 'Weekend']:
        values = df[name].astype(str).str.lower().map({'true': True, 'false': False})
        if values.isna().any():
            raise ValueError(f'{name} must contain only True/False')
        df[name] = values.astype(bool)
    for name in NUMERIC + ['PageValues', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']:
        df[name] = pd.to_numeric(df[name], errors='raise')
        if not np.isfinite(df[name]).all() or (df[name] < 0).any():
            raise ValueError(f'{name} must be finite and nonnegative')
    for name in ['BounceRates', 'ExitRates', 'SpecialDay']:
        if (df[name] > 1).any():
            raise ValueError(f'{name} must be between 0 and 1')
    if df.isna().any().any() or not df.Month.isin(MONTHS).all():
        raise ValueError('Missing values or invalid months')
    if not df.VisitorType.isin(['New_Visitor', 'Returning_Visitor', 'Other']).all():
        raise ValueError('Unknown visitor type')
    for name in ['Administrative', 'Informational', 'ProductRelated', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']:
        if (df[name] % 1 != 0).any():
            raise ValueError(f'{name} must be an integer')
    return df


def clean_data(df):
    # Remove exact records before splitting so copies cannot cross partitions.
    # This is a deterministic rule, not a statistic learned from the holdout.
    return df.drop_duplicates().reset_index(drop=True)


def conversion_table(df, column):
    return df.groupby(column, observed=True).Revenue.agg(sessions='size', purchases='sum', conversion_rate='mean')
