from cytoolz import keyfilter
from itertools import product

import pandas as pd
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
    metadata.index = metadata['column']
    metadata.index.name = None
    return metadata.T
