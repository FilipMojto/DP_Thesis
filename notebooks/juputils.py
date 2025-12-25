import inspect
import pandas as pd
from IPython.display import HTML, display, Code
# def contains_negative(df: pd.DataFrame, col: str) -> bool:
#     """
#     Checks if a specified numeric column in a DataFrame contains at least one negative value.

#     Args:
#         df: The pandas DataFrame to check.
#         col: The name of the column to inspect.

#     Returns:
#         True if the column contains one or more negative values; False otherwise.
#     """
    
#     # 1. Check if the column exists
#     if col not in df.columns:
#         raise ValueError(f"Column '{col}' not found in the DataFrame.")
    
#     # 2. Check if the column is numeric (optional, but good practice)
#     # The comparison (df[col] < 0) will work on non-numeric types but raise a warning
#     # or unexpected results. This check makes it robust.
#     if not pd.api.types.is_numeric_dtype(df[col]):
#         print(f"Warning: Column '{col}' is not a numeric type. Proceeding with comparison.")
    
#     # 3. Core Logic: Check for any value less than 0
#     # Create a boolean series where True indicates a negative value
#     is_negative = (df[col] < 0)
    
#     # .any() returns True if there is at least one True in the boolean series
#     return is_negative.any()

DEF_HTML = HTML("""
<style>
.jp-CodeCell .highlight,
.output pre,
.output code {
    background-color: #111111 !important;
    color: #94d4d4 !important;
}
</style>
""")

def display_func(func, language: str = 'Python', html = DEF_HTML):
    display(html)
    source = inspect.getsource(func)
    display(Code(source, language=language, ))