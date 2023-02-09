from itertools import product

from cytoolz import keyfilter
from more_itertools import windowed
import pyarrow as pa
import pyarrow.compute as pac
from pyarrow import parquet

from dustgoggles.pivot import extract_constants


BRIEF_PQ_META_FIELDS = (
    "physical_type",
    "path_in_schema",
    "compression",
    "encodings",
    "has_dictionary_page",
    "col_ix",
)


def parquet_metadata_records(parquet_fn):
    meta = parquet.read_metadata(parquet_fn)
    records = []
    for col_ix, group_ix in product(
        range(meta.num_columns), range(meta.num_row_groups)
    ):
        record = meta.row_group(group_ix).column(col_ix).to_dict()
        if "statistics" in record.keys():
            record |= record.pop("statistics")
        records.append(record | {"row_group": group_ix, "col_ix": col_ix})
    return records


def _flatten_meta_df(meta_df, extended):
    import pandas as pd

    col_series = []
    for col in meta_df["path_in_schema"].unique():
        col_slice = meta_df.loc[meta_df["path_in_schema"] == col]
        constants, _ = extract_constants(col_slice)
        if extended is False:
            constants = keyfilter(
                lambda k: k in BRIEF_PQ_META_FIELDS, constants
            )
        col_stats = col_slice[
            ["num_values", "total_compressed_size", "total_uncompressed_size"]
        ].sum()
        col_series.append(pd.concat([pd.Series(constants), col_stats]))
    return pd.concat(col_series, axis=1).T


def meta_record_df(meta_records, extended=False):
    import pandas as pd

    meta_df = pd.DataFrame(meta_records)
    if meta_df["path_in_schema"].duplicated().any():
        meta_df = _flatten_meta_df(meta_df, extended)
    meta_df = meta_df.convert_dtypes().rename(
        columns={
            "physical_type": "type",
            "path_in_schema": "column",
            "has_dictionary_page": "dict_page",
            "total_compressed_size": "comp_size",
            "total_uncompressed_size": "uncomp_size",
        }
    )
    if extended is False:
        meta_df = meta_df.reindex(
            columns=[
                "column",
                "type",
                "comp_size",
                "uncomp_size",
                "num_values",
                "dict_page",
                "encodings",
                "compression",
                "col_ix",
            ]
        )
    return meta_df


def meta_column_df(parquet_fn, extended=False):
    metadata = meta_record_df(
        parquet_metadata_records(parquet_fn), extended=extended
    )
    metadata.index = metadata["column"]
    metadata.index.name = None
    return metadata.T


# noinspection PyUnresolvedReferences
def read_between(
    parquet_file, field, bounds, meta_records, columns=None, verbose=True
):
    """
    managed read-between bounds operation, more efficient in some cases
    than parquet.read_table with specified filters
    """
    bound_records = []
    for rec in meta_records:
        if not ((rec["min"] > bounds[1]) or (rec["max"] < bounds[0])):
            bound_records.append(rec)
    if verbose is True:
        print(len(bound_records), len(meta_records))
    if len(bound_records) == 0:
        return
    bound_groups = []

    for rec in bound_records:
        group = parquet_file.read_row_group(rec["row_group"])
        less_than = pac.filter(group, pac.less(group[field], bounds[1]))
        more_than = pac.filter(
            less_than, pac.greater_equal(less_than[field], bounds[0])
        )
        bound_groups.append(more_than)
    return pa.concat_tables(bound_groups)


def sort_chunked(
    input_file,
    output_file,
    sort_column,
    direction="ascending",
    schema=None,
    row_group_size=None,
    use_dictionary=None,
    n_chunks=10,
    inline_processor=None,
    verbose=True,
    version="2.6"
):
    import numpy as np

    write_kwargs, open_kwargs = {}, {}
    if row_group_size is not None:
        write_kwargs["row_group_size"] = row_group_size
    if use_dictionary is not None:
        open_kwargs["use_dictionary"] = use_dictionary
    sort_reader = parquet.ParquetFile(input_file)
    meta_records = parquet_metadata_records(input_file)
    relevant_records = [
        rec for rec in meta_records if rec["path_in_schema"] == sort_column
    ]
    if schema is None:
        schema = sort_reader.read_row_group(0).schema
    sort_writer = parquet.ParquetWriter(
        output_file, version=version, schema=schema, **open_kwargs
    )
    max_value = max([rec["max"] for rec in relevant_records])
    min_value = min([rec["min"] for rec in relevant_records])
    windows = list(windowed(np.linspace(min_value, max_value, n_chunks), 2))
    windows[-1] = (windows[-1][0], windows[-1][1] + 1)
    try:
        for window in windows:
            window_table = read_between(
                sort_reader, sort_column, window, relevant_records, verbose
            )
            if window_table is None:
                continue
            if len(window_table) == 0:
                continue
            if verbose is True:
                print(window, len(window_table))
            window_table = window_table.sort_by([(sort_column, direction)])
            if inline_processor is not None:
                window_table = inline_processor(window_table)
            sort_writer.write_table(window_table, **write_kwargs)
            del window_table
    finally:
        sort_writer.close()
