"""
preprocessing and shorthand functions for dataframes, mappings, and ndarrays.
"""
from functools import reduce
from itertools import product
from operator import attrgetter
import re
from typing import Any, Optional

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
    try:
        constant_indices = df.nunique(axis=axis, dropna=dropna) <= 1
    except TypeError:
        constant_indices = []
        for c in df.columns:
            if isinstance(df[c].iloc[0], list):
                test_series = df[c].map(tuple)
            else:
                test_series = df[c]
            if test_series.nunique(dropna=dropna) <= 1:
                constant_indices.append(c)

    if axis == 0:
        var_indices = [c for c in df.columns if c not in constant_indices]
        constants = df.loc[:, constant_indices]
        variables = df.loc[:, var_indices]
    else:
        var_indices = [c for c in df.indices if c not in constant_indices]
        constants = df.loc[constant_indices]
        variables = df.loc[var_indices]
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
    if isinstance(df.dtypes, np.dtype):
        if re.match("|".join(typenames).lower(), df.dtypes.name.lower()):
            return df
        return pd.DataFrame(index=df.index)
    names = df.dtypes.apply(attrgetter("name")).str.lower()
    type_pattern = f"u?({'|'.join(typenames)})".lower()
    return df[df.dtypes.loc[names.str.match(type_pattern)].index]


def numeric_columns(df):
    """return 'numeric" columns of df."""
    return typed_columns(df, ("int", "float"))


DTYPES = (
    "uint8", "int8", "uint16", "int16",
    "uint32", "int32", "uint64", "int64",
    "float32", "float64"
)


def downcast(arr, atol=0.01, rtol=0.001):
    # NaNs and infs don't count but should
    # be preserved
    nonfinite = ~np.isfinite(arr)
    nonfinite_vals = arr[nonfinite]
    arr[nonfinite] = 0
    recast = None
    for dtype in DTYPES:
        recast = arr.astype(dtype)
        offsets = np.abs(arr - recast)
        if (aerr := np.nanmax(np.abs(offsets))) > atol:
            continue
        if (rerr := np.nanmax(np.abs(offsets / arr))) > rtol:
            continue
        break
    if recast is None:
        raise TypeError
    # noinspection PyUnboundLocalVariable
    return {
        'recast': recast,
        'nonfinite_mask': nonfinite,
        'nonfinite_vals': nonfinite_vals,
        'aerr': aerr,
        'rerr': rerr
    }


def downcast_df(df, atol=0.01, rtol=0.001):
    num = numeric_columns(df)
    cast_series, cast_records = [], []
    for name, col in num.items():
        rec = downcast(col.values.T.copy(), atol, rtol) | {'name': name}
        cast_records.append(rec)
    while len(cast_records) > 0:
        rec = cast_records.pop()
        hasna = rec['nonfinite_mask'].any()
        if hasna and ("int" in (dname := rec['recast'].dtype.name)):
            prefix = "U" if dname.startswith("u") else ""
            depth = (rec['recast'].dtype.itemsize * 8)
            dtype = getattr(pd, f"{prefix}Int{depth}Dtype")()
            series = pd.Series(rec['recast'], name=rec['name'], dtype=dtype)
        else:
            series = pd.Series(rec['recast'], name=rec['name'])
        if hasna:
            series.loc[rec['nonfinite_mask']] = rec['nonfinite_vals']
        cast_series.append(series)
    castdown = pd.concat(list(reversed(cast_series)), axis=1)
    castdown.index = df.index
    # TODO: inefficient inserts
    for c in castdown:
        df[c] = castdown[c]
    return df


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


def unpack(series: pd.Series):
    """unpack nested listlikes to columns"""
    dropped = series.dropna()
    df = pd.DataFrame(dropped.tolist(), index=dropped.index)
    df.columns = [f"{dropped.name}_{i}" for i, _ in enumerate(df.columns)]
    return df


def unpack_column(df: pd.DataFrame, colname):
    """unpack a nested listlike column by name"""
    unpacked = unpack(df[colname])
    df.loc[unpacked.index, unpacked.columns] = unpacked
    return df.drop(columns=colname)


def unique_to_records(df, cols):
    unique_tuples = df[cols].value_counts().index.to_list()
    records = [
        {col: value for col, value in zip(cols, values)}
        for values in unique_tuples
    ]
    # noinspection PyTypeChecker
    return pd.DataFrame.from_dict(records)


def squeeze(series: pd.Series, aggfunc: Optional[str] = None) -> pd.Series:
    """
    removes duplicate index labels from a Series, aggregating values
    associated with each duplicate label using `aggfunc`. if aggfunc is
    None, `squeeze` uses "any" if the Series is boolean and simply returns
    the first element associated with each unique index label if the Series
    is not.
    """
    if aggfunc is None:
        if series.dtype.char == "?":
            aggfunc = "any"
        else:
            return series.loc[~series.index.duplicated()]
    df = pd.DataFrame(series).reset_index()
    ixname = "index" if series.index.name is None else series.index.name
    sname = 0 if series.name is None else series.name
    squeezed = df.pivot_table(
        values=series.name,
        index=ixname,
        aggfunc=aggfunc,
    )[sname]
    squeezed.index.name = series.index.name
    squeezed.name = series.name
    return squeezed
