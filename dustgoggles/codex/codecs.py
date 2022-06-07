import ast
import json
import pickle

from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Optional

from dustgoggles.codex.memutilz import (
    open_block, fetch_block_bytes, exists_block
)
from dustgoggles.func import zero


def json_codec_factory() -> tuple[Callable, Callable]:
    """
    generate a pair of encoder / decoder functions that can be inserted into
    the memorize / remember steps of a mnemonic or used on their own.

    this is a very simple codec that simply uses JSON serialization and
    deserialization. only objects that are stable through JSON serialization
    can/should be passed to this codec.
    """
    def encode(value: Any) -> bytes:
        """return bytestring of json serialization of `value`"""
        return json.dumps(value).encode()

    def decode(blob: bytes) -> Any:
        """return json deserialization of `blob`"""
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


def pickle_codec_factory():
    """
    generate a pair of encoder / decoder functions that can be inserted into
    the memorize / remember steps of a mnemonic or used on their own.

    this just pickles things. It is a useful generic codec that tends to
    produce behaviors that mirror default Python interprocess communication.
    """
    def encode(value: Any) -> bytes:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def decode(blob: bytes) -> Any:
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
        if (exists_ok is False) and exists_block(address):
            raise FileExistsError
        encoded = encode(value)
        size = len(encoded)
        block = open_block(address, size, True, True)
        block.buf[:] = encoded
        return address

    def remember(
        address: str,
        fetch: bool = True,
        decode: Callable[[bytes], Any] = zero
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


def numpy_mnemonic_factory() -> tuple[Callable, Callable]:
    """
    generate a pair of save / load functions that can be used to send and
    retrieve numpy ndarrays to and from shared memory. they can be partially
    evaluated to work as bound methods of higher-level objects, or used
    as standalone objects. unlike generic_mnemonic_factory, these functions
    do not have swappable codecs; they use the array's own buffer protocol.

    TODO: consider implementing inline codecs that pass the buffer /
     memoryview rather than the binary blob in order to support compressing
     / sparsifying / etc. arrays.
    """
    import numpy as np

    def memorize_array(
        array: np.ndarray, address: str, exists_ok: bool, keep_open: bool
    ) -> tuple[dict, Optional[SharedMemory]]:
        """
        generate an array backed by a shared memory block at `address` and
        fill it with the contents of `array`. return a dict containing
        metadata necessary to fetch and reconstruct the array.
        """
        block = open_block(address, array.size * array.itemsize, exists_ok)
        shared_array = np.ndarray(
            array.shape, dtype=array.dtype, buffer=block.buf
        )
        shared_array[:] = array[:]
        metadata = {
            "name": block.name,
            "dtype": array.dtype.str,
            "shape": array.shape,
            "size": array.size * array.itemsize
        }
        if keep_open is False:
            return metadata, None
        return metadata, block

    def remember_array(
        metadata_json: str, fetch: bool = True, copy: bool = True
    ):
        metadata = json.loads(metadata_json)
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
