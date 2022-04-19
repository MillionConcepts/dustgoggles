import json
import pickle

# TODO: deprecated at present
# def string_index_decode(index_buffer):
#     buf = index_buffer.buf
#     index_list = buf.tobytes().decode().strip("\x00").split(",")
#     if index_list == [""]:
#         index_list = []
#     return index_list
#
#
# def string_index_write(index_buffer, new_index):
#     encoded = ",".join(new_index).encode()
#     buf = index_buffer.buf
#     buf[:len(encoded)] = encoded
#     buf[len(encoded):] = b'\x00' * (index_buffer.size - len(encoded))
import sys
from multiprocessing.shared_memory import SharedMemory

from dustgoggles.codex.memutilz import create_block, fetch_block_bytes


def json_pickle_encoder(value):
    try:
        return json.dumps(value).encode()
    except TypeError:
        return pickle.dumps(value, protocol=5)


def json_pickle_decoder(value):
    try:
        decoded = value.decode()
        return json.loads(decoded)
    except UnicodeDecodeError:
        return pickle.loads(value)


def memorize_generic(value, address, encoder, exists_ok):
    encoded = encoder(value)
    size = len(encoded)
    block = create_block(address, size, exists_ok)
    block.buf[:] = encoded
    if "win" not in sys.platform:
        block.close()
    return address


def remember_generic(address, decoder):
    stream = fetch_block_bytes(address)
    if stream is None:
        return stream
    return decoder(stream)
