import pandas as pd


def load_csv(uploaded_file):
    """Load uploaded CSV file into a pandas DataFrame."""
    return pd.read_csv(uploaded_file)


def get_preview(df, rows=5):
    """Return first few rows for quick dataset preview."""
    return df.head(rows)


def load_and_clean_csv(
    file_path,
    missing_strategy="fill",
    fill_value=0,
    drop_axis=0,
):
    """
    Load, inspect, and clean a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
        missing_strategy (str): "fill" to fill missing values, "drop" to drop them.
        fill_value (any): Value used when missing_strategy="fill".
        drop_axis (int): 0 to drop rows with nulls, 1 to drop columns with nulls.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Null values per column:")
    print(df.isnull().sum())

    if missing_strategy == "fill":
        cleaned_df = df.fillna(fill_value)
    elif missing_strategy == "drop":
        cleaned_df = df.dropna(axis=drop_axis)
    else:
        raise ValueError("missing_strategy must be either 'fill' or 'drop'")

    return cleaned_df


def clean_dataframe(df, missing_strategy="fill", fill_value=0, drop_axis=0):
    """
    Clean an existing DataFrame by handling missing values.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        missing_strategy (str): "fill" or "drop".
        fill_value (any): Value used when strategy is "fill".
        drop_axis (int): 0 -> drop rows, 1 -> drop columns.
    """
    if missing_strategy == "fill":
        return df.fillna(fill_value)
    if missing_strategy == "drop":
        return df.dropna(axis=drop_axis)
    raise ValueError("missing_strategy must be either 'fill' or 'drop'")


def get_basic_info(df):
    """Return basic dataset info for UI display."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "null_values": df.isnull().sum().to_dict(),
    }


def get_basic_stats(df, column=None):
    """
    Get mean, median, and mode.

    If column is provided, returns stats for that column.
    If column is None, returns stats for all numeric columns.
    """
    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column '{column}' must be numeric")

        mode_values = series.mode().tolist()
        return {
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "mode": mode_values,
        }

    numeric_df = df.select_dtypes(include="number")
    return {
        "mean": numeric_df.mean().to_dict(),
        "median": numeric_df.median().to_dict(),
        "mode": numeric_df.mode().iloc[0].to_dict() if not numeric_df.empty else {},
    }


def groupby_average(df, group_col, target_cols=None):
    """
    Group by one column and calculate averages.

    Parameters:
        group_col (str): Column used for grouping.
        target_cols (list[str] or None): Numeric columns to average.
                                       If None, all numeric columns are used.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe")

    if target_cols is None:
        target_cols = df.select_dtypes(include="number").columns.tolist()
    else:
        missing_cols = [col for col in target_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Target columns not found: {missing_cols}")

    return df.groupby(group_col)[target_cols].mean().reset_index()


def get_min_max(df, column=None):
    """
    Find min and max values.

    If column is provided, returns min and max for that column.
    If column is None, returns min and max for all numeric columns.
    """
    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column '{column}' must be numeric")

        return {"column": column, "min": series.min(), "max": series.max()}

    numeric_df = df.select_dtypes(include="number")
    return {
        "min": numeric_df.min().to_dict(),
        "max": numeric_df.max().to_dict(),
    }


def describe_dataset(df, include="all"):
    """
    Return dataset description using pandas describe.

    include:
      - "all" for all columns
      - "number" for numeric columns
    """
    return df.describe(include=include)
