import ast
import json
from multiprocessing.shared_memory import SharedMemory
import pickle
from typing import Any, Callable, Mapping

from dustgoggles.codex.memutilz import create_block, fetch_block_bytes


def json_codec_factory() -> tuple[Callable, Callable]:
    """
    generate a pair of encoder / decoder functions that can be inserted into
    the memorize / remember steps of a mnemonic or used on their own.

    this is a very simple codec that simply uses JSON serialization and
    deserialization. only objects that are stable through JSON serialization
    can/should be passed to this codec.
    """
    def encode(value: Any) -> bytes:
        return json.dumps(value).encode()

    def decode(blob: bytes) -> Any:
        return json.loads(blob)
    return encode, decode


def ast_codec_factory():
    """
    generate a pair of encoder / decoder functions that can be inserted into
    the memorize / remember steps of a mnemonic or used on their own.

    this is a very simple codec that uses the Python abstract syntax tree
    serializer and deserializer. It will reliably serialize a wider variety of
    Python literals than using JSON serialization, but is much slower.
    """
    def encode(value: Any) -> bytes:
        return str(value).encode()

    def decode(blob: bytes) -> Any:
        return ast.literal_eval(blob.decode())
    return encode, decode


def json_pickle_codec_factory():
    """
    generate a pair of encoder / decoder functions that can be inserted into
    the memorize / remember steps of a mnemonic or used on their own.

    this codec attempts to serialize passed objects as JSON. failing that,
    it will attempt to serialize them using pickle. This is a useful generic
    codec that tends to produce behaviors that mirror default Python
    interprocess communication.
    """
    def encode(value: Any) -> bytes:
        try:
            return json.dumps(value).encode()
        except TypeError:
            return pickle.dumps(value, protocol=5)

    def decode(blob: bytes) -> Any:
        try:
            return json.loads(blob.decode())
        except UnicodeDecodeError:
            return pickle.loads(blob)

    return encode, decode


def generic_mnemonic_factory():
    """
    generate a pair of save / load functions that can be used to send and
    retrieve objects from shared memory. this is a general-purpose mnemonic
    that will work for many sorts of objects. these save / load functions can
    be partially evaluated to create bound methods (see objects in
    codex.implements). alternatively, different encoders, decoders, address
    schemes, etc. can be dynamically passed into these functions at runtime.
    """

    def memorize(
        value: Any,
        address: str,
        exists_ok: bool,
        encode: Callable[[Any], bytes]
    ) -> str:
        """
        serialize `value` using `encode` and place it into a shared memory
        block at `address`. If a block already exists at `address` and
        exists_ok is not True, will raise a FileExistsError. returns
        `address`.
        """
        encoded = encode(value)
        size = len(encoded)
        block = create_block(address, size, exists_ok)
        block.buf[:] = encoded
        return address

    def remember(
        address: str,
        decode: Callable[[bytes], Any],
        fetch: bool = True
    ) -> Any:
        """
        open the shared memory block at `address`. if `fetch` is False,
        return the block itself. otherwise, copy the bytes into memory,
        decode them with `decode`, and return the result.
        """
        if fetch is False:
            return SharedMemory(name=address)
        return decode(fetch_block_bytes(address))

    return memorize, remember


# there's no point to having swappable codecs in this mnemonic: we use the
# array's buffer protocol.
def numpy_mnemonic_factory() -> tuple[Callable, Callable]:
    """
    """
    import numpy as np

    def memorize_array(
        array: np.ndarray, address: str, exists_ok: bool
    ) -> dict:
        block = create_block(address, array.size * array.itemsize, exists_ok)
        shared_array = np.ndarray(
            array.shape, dtype=array.dtype, buffer=block.buf
        )
        shared_array[:] = array[:]
        return {
            "name": block.name,
            "dtype": array.dtype.str,
            "shape": array.shape,
            "size": array.size * array.itemsize
        }

    def remember_array(
        metadata: Mapping, fetch: bool = True, copy: bool = True
    ):
        block = SharedMemory(name=metadata["name"])
        if fetch is False:
            return block
        array = np.ndarray(
            metadata['shape'], dtype=metadata['dtype'], buffer=block.buf
        )
        if copy is True:
            return array.copy(), None
        return array, block

    return memorize_array, remember_array


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
