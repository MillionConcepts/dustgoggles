"""
preprocessing and shorthand functions for dataframes, mappings, and ndarrays.
"""
import warnings
from functools import reduce
from itertools import product
from numbers import Number, Real
from operator import attrgetter
import re
from typing import Any, Optional, Union, Sequence, Literal

from cytoolz import valfilter
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

from dustgoggles.func import gmap

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
    df: pd.DataFrame,
    to_dict=True,
    drop_constants=False,
    how="rows",
    dropna=True
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
    if how not in ("rows", "columns"):
        raise ValueError(f"unknown how {how}")
    if how == "rows":
        constant_indices = pd.Series(False, df.columns)
    else:
        constant_indices = pd.Series(False, df.index)
    try:
        method = "iterrows" if how == "columns" else "items"
        for ix, series in getattr(df, method)():
            series = series if dropna is False else series.dropna()
            if (len(series) == 0) or (series == series.iloc[0]).all():
                constant_indices.loc[ix] = True
    except TypeError:
        # TODO: still necessary?
        if how == "columns":
            raise NotImplementedError("nested fields not supported columnwise")
        for c in df.columns:
            if isinstance(df[c].iloc[0], list):
                test_series = df[c].map(tuple)
            else:
                test_series = df[c]
            if (test_series.iloc[0] == test_series).all():
                constant_indices.loc[c] = True
    if how == "rows":
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


def check_in_dtype_bounds(
    amin: Union[int, float], amax: Union[int, float], dt: np.dtype
):
    info = np.iinfo(dt) if dt.kind in 'iu' else np.finfo(dt)
    try:
        return bool(amin >= info.min and amax <= info.max)
    except TypeError:  # just too big!
        return False


def _maybeseries(arr, dtype, library, name, aerr, rerr):
    arr = arr.copy() if arr.dtype == dtype else arr.astype(dtype)
    if library == 'pandas':
        arr = pd.Series(arr, name=name)
    return {'recast': arr, 'aerr': aerr, 'rerr': rerr}


def _downcast_int_array(
    arr: np.ndarray,
    library: Literal["numpy", "pandas"],
    name: Optional[str] = None
):
    amin, amax = arr.min(), arr.max()
    sign = 'u' if amin >= 0 else ''
    for depth in (8, 16, 32, 64):
        dt = np.dtype(f'{sign}int{depth}')
        if check_in_dtype_bounds(amin, amax, dt):
            # there's never 'error' casting int types to int types
            return _maybeseries(arr, dt, library, name, 0, 0)
    raise TypeError("Unusual failure in downcasting integer array.")


def downcast(
    arr: Sequence[Number],
    atol: float = 5e-5,
    rtol: float = 1e-6,
    library: Literal["numpy", "pandas"] = "numpy",
    use_nillable_int: bool = False
):
    """
    NOTE: doesn't currently handle nullable integer types, structured types,
        complex numbers, object types, etc.
    """
    if use_nillable_int is True and library == 'numpy':
        raise TypeError("Can't generate numpy dtypes with nillable ints. ")
    name = arr.name if hasattr(arr, "name") and library == "pandas" else None
    arr = np.asarray(arr)
    if arr.dtype.kind in 'cMO':
        raise TypeError(
            f"This function doesn't handle {arr.dtype} arrays. Cast to an "
            f"integer or real type first if you want to use this function."
        )
    if arr.dtype.kind in 'ui':
        return _downcast_int_array(arr, library, name)
    arr_finite = arr[np.isfinite(arr)]
    if len(arr_finite) == 0:
        # there's no meaningful way to recast in this case
        return _maybeseries(arr, arr.dtype, library, name, 0, 0)
    has_nonfinite = arr_finite.size < arr.size
    nzero_mask = arr_finite != 0
    arr_nzero = arr_finite[nzero_mask]
    if arr_nzero.size == 0 and not has_nonfinite:
        return _maybeseries(arr, np.uint8, library, name, 0, 0)
    elif arr_nzero.size == 0 and use_nillable_int is False:
        return arr.astype(np.float16)
    elif arr_nzero.size == 0:
        return _maybeseries(arr, pd.UInt8Dtype(), library, name, 0, 0)
    amin, amax = arr_finite.min(), arr_finite.max()
    for kind, number in product(
        ('uint', 'int', 'float'), (8, 16, 32, 64, 128)
    ):
        # we're done checking if this next clause is True
        if kind == 'float' and number == arr.dtype.itemsize * 8:
            return _maybeseries(arr, arr.dtype, library, name, 0, 0)
        elif kind != 'float' and number == 128:
            continue
        elif kind != 'float' and has_nonfinite and use_nillable_int is False:
            continue
        elif kind == 'float' and number == 8:
            continue
        dtype = np.dtype(f"{kind}{number}")
        if check_in_dtype_bounds(amin, amax, dtype) is False:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            recast = arr_nzero.astype(dtype)
        offsets = np.abs(arr_nzero - recast)
        if (aerr := np.nanmax(np.abs(offsets))) > atol:
            continue
        if (rerr := np.nanmax(np.abs(offsets / arr_nzero))) > rtol:
            continue
        if kind in ('uint', 'int') and has_nonfinite:
            # we've already checked to make sure we're returning this as pandas
            dtype = getattr(
                pd, f"{kind[0].upper()}{kind[1:].capitalize}{number}Dtype"
            )()
        return _maybeseries(arr, dtype, library, name, aerr, rerr)
    raise TypeError("Unusual error in recasting this array.")


def downcast_df(
    df: pd.DataFrame,
    atol: float = 1e-5,
    rtol: float = 1e-7,
    use_nillable_int: bool = False,
    reassemble: bool = True,
    copy_all: bool = True
):
    cast_series = []
    for _, col in df.items():
        if not is_numeric_dtype(col.dtype):
            cast_series.append(col if copy_all is False else col.copy())
            continue
        cast_series.append(
            downcast(col, atol, rtol, 'pandas', use_nillable_int)['recast']
        )
    if reassemble is True:
        return pd.concat(cast_series, axis=1)
    return cast_series
    # cast_series, cast_records = [], []
    # for name, col in num.items():
    #     rec = downcast(col.values.copy(), atol, rtol) | {'name': name}
    #     cast_records.append(rec)
    # while len(cast_records) > 0:
    #     rec = cast_records.pop()
    #     hasna = rec['nonfinite_mask'].any()
    #     if hasna and ("int" in (dname := rec['recast'].dtype.name)):
    #         prefix = "U" if dname.startswith("u") else ""
    #         depth = (rec['recast'].dtype.itemsize * 8)
    #         dtype = getattr(pd, f"{prefix}Int{depth}Dtype")()
    #         series = pd.Series(rec['recast'], name=rec['name'], dtype=dtype)
    #     else:
    #         series = pd.Series(rec['recast'], name=rec['name'])
    #     if hasna:
    #         series.loc[rec['nonfinite_mask']] = rec['nonfinite_vals']
    #     cast_series.append(series)
    # castdown = pd.concat(list(reversed(cast_series)), axis=1)
    # castdown.index = df.index
    # # TODO: inefficient inserts
    # for c in castdown:
    #     df[c] = castdown[c]
    # return df


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
