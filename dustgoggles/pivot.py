"""
preprocessing and shorthand functions for dataframes, mappings, and ndarrays.
"""

import numpy as np
import pandas as pd


def itemize_numpy(obj):
    """
    convert objects of numpy dtypes to python scalars. in this context,
    primarily for json serialization.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dupe_df_block(dataframe, rows_to_repeat):
    return pd.DataFrame(
        np.repeat(dataframe.values, rows_to_repeat, axis=0),
        columns=dataframe.columns,
    )


def check_and_drop_duplicate_columns(dataframe):
    extra_columns = dataframe.columns[dataframe.columns.duplicated()]
    if len(extra_columns) == 0:
        return dataframe
    for column in extra_columns:
        test_equality = (
            dataframe.loc[:, column] == dataframe.loc[:, column].iloc[0, 0]
        )
        assert test_equality.all(axis=None)
    return dataframe.loc[:, ~dataframe.columns.duplicated()]


def extract_constants(df, to_dict=True, drop_constants=False):
    constant_columns = df.nunique() == 1
    constants = df.loc[:, constant_columns]
    variables = df.loc[:, ~constant_columns]
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
