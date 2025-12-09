import pandas as pd

def contains_negative(df: pd.DataFrame, col: str) -> bool:
    """
    Checks if a specified numeric column in a DataFrame contains at least one negative value.

    Args:
        df: The pandas DataFrame to check.
        col: The name of the column to inspect.

    Returns:
        True if the column contains one or more negative values; False otherwise.
    """
    
    # 1. Check if the column exists
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")
    
    # 2. Check if the column is numeric (optional, but good practice)
    # The comparison (df[col] < 0) will work on non-numeric types but raise a warning
    # or unexpected results. This check makes it robust.
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Warning: Column '{col}' is not a numeric type. Proceeding with comparison.")
    
    # 3. Core Logic: Check for any value less than 0
    # Create a boolean series where True indicates a negative value
    is_negative = (df[col] < 0)
    
    # .any() returns True if there is at least one True in the boolean series
    return is_negative.any()