"""

lightning_hydra_classifiers/utils/pandas_utils.py

Author: Jacob A Rose
Created: Saturday Sept 18th, 2021

Help functions for common pandas tasks


"""

import logging
import pandas as pd

__all__ = ["optimize_dtypes"]


def optimize_dtypes(df: pd.DataFrame,
                    convert2categorical_ratio: float = 0.5,
                    verbose: bool=False) -> pd.DataFrame:
    """
    Optimize dtypes of a pandas dataframe by:
        1. First applying builtin pd.DataFrame.convert_dtypes()
        2. Then selectively converting columns to categorical based on the ratio of unique samples to total samples.
    
    Returns: dtype-optimized dataframe
    """
#     columns = [col for col in df.columns if col not in exclude_cols]
    df = df.convert_dtypes()
    num_rows = df.shape[0]    
    for col in df.columns:
        num_unique = df[col].nunique()
        ratio = (num_unique/num_rows)
        
        if ratio <= convert2categorical_ratio:
            df[col] = df[col].astype('category')
            if verbose:
                print(f"Column: {col}->categorical, N_unique ratio: {ratio:.2f}")
    return df