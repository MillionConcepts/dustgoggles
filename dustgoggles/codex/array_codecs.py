import sys
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from dustgoggles.structures import dig_and_edit


def reference_shared_memory_arrays(
    block_info, fetch=True
) -> tuple[dict, dict]:
    blocks = {
        variable: SharedMemory(name=info["name"])
        for variable, info in block_info.items()
    }
    if fetch is True:
        chunk = {
            variable: np.ndarray(
                info["shape"], dtype=info["dtype"], buffer=blocks[variable].buf
            )
            for variable, info in block_info.items()
        }
    else:
        chunk = {}
    return blocks, chunk

#
#
# def share_mapping(mapping, codec=share_array):
#     block_info = {}
#
#     for key, value in mapping.items():
#         block_info[key] =
#     return block_info

#
# def slice_into_shared_chunks(chunksz, data, nphots):
#     variable_names = [key for key in data.keys()]
#     chunk_slices = make_chunk_slices(chunksz, nphots)
#     total_chunks = len(chunk_slices)
#     block_directory = {}
#     for chunk_ix in range(total_chunks):
#         block_directory = slice_chunk_into_memory(
#             block_directory, chunk_ix, data, chunk_slices, variable_names
#         )
#     return block_directory
#

def make_chunk_slices(chunksz, nphots):
    table_indices = []
    total_chunks = range(int(nphots / chunksz) + 1)
    for chunk_ix in total_chunks:
        chunkbeg, chunkend = chunk_ix * chunksz, (chunk_ix + 1) * chunksz
        if chunkend > nphots:
            chunkend = None
        table_indices.append((chunkbeg, chunkend))
    return table_indices
#
#
# def slice_into_memory(data, indices):
#     return send_to_shared_memory(
#         {key: value[slice(*indices)] for key, value in data.items()}
#     )


# def slice_chunk_into_memory(
#     block_directory, chunk_ix, data, table_indices, variable_names=None
# ):
#     arrays = [
#         array[slice(*table_indices[chunk_ix])] for array in data.values()
#     ]
#     if variable_names is None:
#         variable_names = tuple(data.keys())
#     block_info = send_to_shared_memory(
#         {
#             variable_name: array
#             for variable_name, array in zip(variable_names, arrays)
#         }
#     )
#     block_directory[chunk_ix] = block_info
#     return block_directory


def send_mapping_to_shared_memory(
    mapping,
    filter_func=None,
    setter_func=lambda k, v: memorize_array(v)
):
    block_directory = dig_and_edit(
        mapping, filter_func, setter_func
    )




    return block_directory


def unlink_nested_block_dict(cal_data):
    all_cal_blocks = []
    for cal_name, cal_info in cal_data.items():
        if cal_info is None:
            continue
        cal_blocks, _ = reference_shared_memory_arrays(cal_info, fetch=False)
        all_cal_blocks += list(cal_blocks.values())
    for block in all_cal_blocks:
        block.close()
        block.unlink()


def get_column_from_shared_memory(results, column_name, unlink=True):
    column_info = {
        chunk_ix: results[chunk_ix][column_name]
        for chunk_ix in sorted(results.keys())
    }
    blocks, column_slices = reference_shared_memory_arrays(column_info)
    column = np.hstack(list(column_slices.values()))
    del column_slices
    if unlink is True:
        for block in blocks.values():
            block.close()
            block.unlink()
    return column

