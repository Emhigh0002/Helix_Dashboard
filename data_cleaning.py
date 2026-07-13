import pandas as pd
import numpy as np

def clean_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows from the DataFrame."""
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates()
    removed = initial_rows - len(df_cleaned)
    return df_cleaned, removed

def clean_missing_values(df: pd.DataFrame, strategy: str, columns: list[str] = None, fill_value=None) -> tuple[pd.DataFrame, dict]:
    """
    Handle missing values.
    Strategies:
    - 'drop': Drop rows with any missing values in the specified columns (or all columns).
    - 'mean': Impute missing numerical values with the mean.
    - 'median': Impute missing numerical values with the median.
    - 'mode': Impute missing values with the mode (useful for categorical/numerical).
    - 'constant': Impute missing values with a user-defined fill_value.
    """
    df_cleaned = df.copy()
    if not columns:
        columns = df_cleaned.columns.tolist()
        
    summary = {}
    for col in columns:
        missing_count = int(df_cleaned[col].isna().sum())
        if missing_count == 0:
            continue
            
        summary[col] = {"missing_before": missing_count}
        
        if strategy == 'drop':
            df_cleaned = df_cleaned.dropna(subset=[col])
            summary[col]["action"] = "dropped_rows"
            summary[col]["affected_rows"] = missing_count
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                mean_val = df_cleaned[col].mean()
                df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                summary[col]["action"] = "imputed_mean"
                summary[col]["value"] = float(mean_val)
            else:
                summary[col]["action"] = "skipped (non-numeric)"
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
                summary[col]["action"] = "imputed_median"
                summary[col]["value"] = float(median_val)
            else:
                summary[col]["action"] = "skipped (non-numeric)"
        elif strategy == 'mode':
            mode_series = df_cleaned[col].mode()
            if not mode_series.empty:
                mode_val = mode_series[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                summary[col]["action"] = "imputed_mode"
                summary[col]["value"] = str(mode_val)
            else:
                summary[col]["action"] = "skipped (no mode)"
        elif strategy == 'constant':
            df_cleaned[col] = df_cleaned[col].fillna(fill_value if fill_value is not None else "")
            summary[col]["action"] = "imputed_constant"
            summary[col]["value"] = str(fill_value)
            
    return df_cleaned, summary

def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> tuple[pd.Series, float, float]:
    """Detect outliers in a numeric column using the IQR method."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")
        
    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - k * iqr
    upper_bound = q75 + k * iqr
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers, lower_bound, upper_bound

def handle_outliers(df: pd.DataFrame, column: str, method: str = 'cap', k: float = 1.5) -> tuple[pd.DataFrame, int]:
    """
    Handle outliers in a numeric column.
    Methods:
    - 'drop': Drop rows containing outliers.
    - 'cap': Cap outliers at the upper and lower bounds.
    """
    df_cleaned = df.copy()
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df_cleaned, column, k)
    outliers_count = int(outliers.sum())
    
    if outliers_count == 0:
        return df_cleaned, 0
        
    if method == 'drop':
        df_cleaned = df_cleaned[~outliers]
    elif method == 'cap':
        df_cleaned[column] = df_cleaned[column].clip(lower=lower_bound, upper=upper_bound)
        
    return df_cleaned, outliers_count

def scale_features(df: pd.DataFrame, columns: list[str], method: str = 'standard') -> pd.DataFrame:
    """Scale numeric columns using 'standard' (z-score) or 'minmax' scaling."""
    df_scaled = df.copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df_scaled[col]):
            continue
        if method == 'standard':
            mean = df_scaled[col].mean()
            std = df_scaled[col].std()
            if std != 0:
                df_scaled[col] = (df_scaled[col] - mean) / std
        elif method == 'minmax':
            min_val = df_scaled[col].min()
            max_val = df_scaled[col].max()
            diff = max_val - min_val
            if diff != 0:
                df_scaled[col] = (df_scaled[col] - min_val) / diff
    return df_scaled

def convert_column_type(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    """Convert a column to a target data type ('numeric', 'categorical', 'datetime')."""
    df_converted = df.copy()
    if target_type == 'numeric':
        df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
    elif target_type == 'categorical':
        df_converted[column] = df_converted[column].astype(str)
    elif target_type == 'datetime':
        df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
    return df_converted
