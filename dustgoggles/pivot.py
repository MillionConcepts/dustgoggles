"""
preprocessing and shorthand functions for dataframes, mappings, and ndarrays.
"""
from typing import Any

import numpy as np
import pandas as pd


def itemize_numpy(obj: Any):
    """
    convert objects of numpy dtypes to python scalars. in this context,
    primarily for json serialization.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dupe_df_block(dataframe: pd.DataFrame, repeats: int):
    """
    create a new dataframe by duplicating "dataframe" "repeats" times
    """
    return pd.DataFrame(
        np.repeat(dataframe.values, repeats, axis=0),
        columns=dataframe.columns,
    )


# TODO, maybe: allow this to work with columns that have collections
#  in individual cells
def check_and_drop_duplicate_columns(dataframe: pd.DataFrame):
    """
    drop duplicate columns -- in both name and value -- from a dataframe.
    if equal-named columns are _not_ equal-valued, raises a ValueError.
    """
    extra_columns = dataframe.columns[dataframe.columns.duplicated()]
    if len(extra_columns) == 0:
        return dataframe
    for column in extra_columns:
        test_equality = (dataframe.loc[:, column] == dataframe.loc[:, column])
        if not test_equality.all(axis=None):
            raise ValueError
    return dataframe.loc[:, ~dataframe.columns.duplicated()]


def extract_constants(df, to_dict=True, drop_constants=False, how='rows'):
    """
    extract 'constant' values from a dataframe -- by default, columns with
    the same value in each row; if how == 'columns', then indices with the
    same value in each column. if to_dict is True, transforms them into a
    dictionary with keys equal to column/index names and values equal to the
    constant values. if drop_constants is True, also removes these
    constant-valued rows/columns from the returned dataframe.
    returns the constants (as a dict if specified) and either the
    variable-only or full original dataframe (as specified)
    """
    if how == 'rows':
        axis = 0
    elif how == 'columns':
        axis = 1
    else:
        raise ValueError(f"unknown how {how}")
    constant_indices = df.nunique(axis=axis) == 1
    if axis == 0:
        constants = df.loc[:, constant_indices]
        variables = df.loc[:, ~constant_indices]
    else:
        constants = df.loc[constant_indices]
        variables = df.loc[~constant_indices]
    if to_dict:
        constants = constants.iloc[0].to_dict()
    if drop_constants:
        return constants, variables
    return constants, df


def split_on(
    df: pd.DataFrame, predicate: pd.Series
) -> [pd.DataFrame, pd.DataFrame]:
    return df.loc[predicate], df.loc[~predicate]


def pdstr(str_method_name, *str_args, **str_kwargs):
    """
    creates a mappable function that accesses .str methods of passed Series
    """

    def replacer(series: pd.Series):
        method = getattr(series.str, str_method_name)
        return method(*str_args, **str_kwargs)

    return replacer
