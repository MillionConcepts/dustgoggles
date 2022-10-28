"""
preprocessing and shorthand functions for dataframes, mappings, and ndarrays.
"""
from functools import reduce
from operator import attrgetter
from typing import Any

from cytoolz import valfilter
import numpy as np
import pandas as pd

from dustgoggles.structures import listify


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
        test_equality = dataframe.loc[:, column] == dataframe.loc[:, column]
        if not test_equality.all(axis=None):
            raise ValueError
    return dataframe.loc[:, ~dataframe.columns.duplicated()]


def extract_constants(
    df, to_dict=True, drop_constants=False, how="rows", dropna=True
):
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
    if how == "rows":
        axis = 0
    elif how == "columns":
        axis = 1
    else:
        raise ValueError(f"unknown how {how}")
    constant_indices = df.nunique(axis=axis, dropna=dropna) == 1
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


def typed_columns(df, typenames):
    """return columns of df of types that match patterns given in typenames."""
    if isinstance(typenames, str):
        typenames = (typenames,)
    names = df.dtypes.apply(attrgetter("name"))
    return df[df.dtypes.loc[names.str.match("|".join(typenames))].index]


def numeric_columns(df):
    """return 'numeric" columns of df."""
    return typed_columns(df, ("int", "float"))


def demote(df, dtype, typenames=("int", "float")):
    """
    cast columns of df matching typenames to dtype.
    return worst-case absolute and relative errors along with demoted df.
    """
    candidates = typed_columns(df, typenames)
    demoted = candidates.astype(dtype)
    offsets = (candidates - demoted).abs()
    a_err = offsets.abs().max()
    r_err = (offsets / candidates).abs().max()
    return demoted, a_err, r_err


def check_demote(df, dtype, typenames=("int", "float"), rtol=0.001, atol=0.01):
    demoted, a_err, r_err = demote(df, dtype, typenames)
    a_ok = a_err.loc[a_err < atol]
    r_ok = r_err.loc[r_err < rtol]
    return demoted[list(set(a_ok.index).intersection(set(r_ok.index)))]


def junction(df1, df2, columns, set_method="difference"):
    keys1 = df1[list(columns)].value_counts().index.to_list()
    keys2 = df2[list(columns)].value_counts().index.to_list()
    return getattr(set(keys1), set_method)(set(keys2))


def smash(df, by, values=None):
    by = listify(by)
    if values is not None:
        df = df.loc[df[by].isin(values).all(axis=1)]
    df = df.melt(by)
    names = reduce(
        lambda x, y: x + "_" + y,
        [v for _, v in df[by + ["variable"]].astype(str).items()],
    )
    return pd.Series(df["value"].to_numpy(), index=names)


def categorizable(df, threshold=255):
    uniques = {}
    for c in df.columns:
        try:
            uniques[c] = len(df[c].unique())
        except TypeError:
            continue
    categories = valfilter(lambda v: v <= threshold, uniques)
    return list(categories.keys())


def categorize(df, threshold=255):
    cat_columns = categorizable(df, threshold)
    columns = [
        df[c]
        if c not in cat_columns
        else pd.Series(pd.Categorical(df[c]), name=c)
        for c in df.columns
    ]
    return pd.concat(columns, axis=1)


def unique_to_records(df, cols):
    unique_tuples = df[cols].value_counts().index.to_list()
    records = [
        {col: value for col, value in zip(cols, values)}
        for values in unique_tuples
    ]
    # noinspection PyTypeChecker
    return pd.DataFrame.from_dict(records)
