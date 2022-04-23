from abc import abstractmethod, ABC
import atexit
import time
from functools import partial
from multiprocessing.shared_memory import ShareableList, SharedMemory
from random import randint, randbytes
from typing import Any, Union, Optional, Callable

from dustgoggles.codex.codecs import (
    json_pickle_codec_factory, generic_mnemonic_factory,
    numpy_mnemonic_factory, json_codec_factory, ast_codec_factory
)
from dustgoggles.codex.memutilz import create_block, fetch_block_bytes, \
    exists_block, delete_block


class LockoutTagout:
    def __init__(self, name=None):
        if name is None:
            name = randint(100000, 999999)
        self.name = name
        self.tag = randbytes(5)
        self.lock = create_block(
            f"{self.name}_lock", exists_ok=True, size=5
        )

    def acquire(self, timeout=0.1, increment=0.0001):
        acquired = False
        time_waiting = 0
        while acquired is False:
            tag = self.lock.buf.tobytes()
            if tag == self.tag:
                acquired = True
            elif tag == b"\x00\x00\x00\x00\x00":
                self.lock.buf[:] = self.tag
            # the extra loop here is for a possibly-unnecessary double check
            # that someone else didn't write their key to the lock at the
            # exact same time.
            else:
                time_waiting += increment
                if time_waiting > timeout:
                    raise TimeoutError("timed out waiting for lock")
                time.sleep(increment)

    def release(self, release_only_mine=True):
        tag = self.lock.buf.tobytes()
        if (tag != self.tag) and (release_only_mine is True):
            raise ConnectionRefusedError
        self.lock.buf[:] = b"\x00\x00\x00\x00\x00"


class ShareableIndex(ABC):

    def __init__(self, name=None, create=False, cleanup_on_exit=True, **_):
        if create is False:
            if not exists_block(name):
                raise FileNotFoundError(
                    f"Shared memory block {name} does not exist and "
                    f"create=False passed. Construct this index with "
                    f"create=True if you want to initialize a new index."
                )
        self.loto = LockoutTagout(name)
        # if SharedMemory objects are instantiated in __main__,
        # multiprocessing.util._exit_function() generally does a good job
        # of cleaning them up. however, blocks created in child processes
        # will often not be cleanly and automatically unlinked. hence:
        if cleanup_on_exit is True:
            atexit.register(self.close)

    @abstractmethod
    def update(self, sync):
        pass

    @abstractmethod
    def add(self, key, value=None):
        pass

    @abstractmethod
    def __delitem__(self, key):
        pass

    @abstractmethod
    def close(self):
        pass

    def remove(self, key):
        return self.__delitem__(key)


class DictIndex(ShareableIndex):
    """
    very simple index based on dumping json into a shared memory block.

    note that object keys must be strings -- integers, floats, etc.
    will be converted to strings by json serialization / deserialization.
    """
    def __init__(self, name=None, create=False, cleanup_on_exit=True):
        super().__init__(name, create, cleanup_on_exit)
        memorize, remember = generic_mnemonic_factory()
        encode, decode = json_codec_factory()
        self.name = name
        self.memorize = partial(
            memorize, address=self.name, exists_ok=True, encode=encode
        )
        self.remember = partial(
            remember, metadata=self.name, fetch=True, decode=decode
        )
        self._cache = {}
        if create is True:
            self.memorize({})
            self.memory = SharedMemory(self.name)

    def update(self, sync=True):
        if sync is True:
            self._cache = self.remember()
        return self._cache

    def keys(self):
        return self.update().keys()

    def values(self):
        return self.update().values()

    def __getitem__(self, key):
        return self.update()[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        self.loto.acquire()
        dictionary = self.update()
        dictionary[key] = value
        self.memorize(dictionary)
        self.loto.release()
        self.memory = SharedMemory(self.name)

    def add(self, key, value = None):
        self[key] = value

    def __delitem__(self, key):
        self.loto.acquire()
        dictionary = self.update()
        del dictionary[key]
        self.memorize(dictionary)
        self.loto.release()
        self.memory = SharedMemory(self.name)

    def close(self):
        for block in (self.loto.lock, SharedMemory(self.name)):
            block.unlink()
            block.close()
        atexit.unregister(self.close)


class ListIndex(ShareableIndex):
    """
    limited but fast index based on multiprocessing.shared_memory.ShareableList
    """
    def __init__(
        self,
        name=None,
        create=False,
        cleanup_on_exit=True,
        length=64,
        max_characters=64,
    ):
        super().__init__(name, create, cleanup_on_exit)
        if create is False:
            self.memory = ShareableList(name)
        else:
            self.memory = ShareableList(
                [b"\x00" * max_characters for _ in range(length)], name=name
            )
        self._cache = []
        self.length = 0

    def update(self, sync=True):
        if sync is True:
            # TODO: optimize all this stuff with lookahead and lookbehind
            #  for speed, etc.
            self._cache = []
            ix = 0
            for ix, key in enumerate(self.memory):
                if key == b"":
                    break
                self._cache.append(key)
            self.length = ix
        return tuple(self._cache)

    def add(self, key, value=None):
        if value is not None:
            raise TypeError(
                f"{type(self)} does not support assigning values to keys."
            )
        self.loto.acquire()
        self.update()
        self.memory[self.length] = key
        self.loto.release()

    def __delitem__(self, key):
        self.loto.acquire()
        self.update()
        iterator = enumerate(self.memory)
        ix, item = None, None
        while item != key:
            try:
                ix, item = next(iterator)
            except StopIteration:
                raise KeyError(f"{key} not found in this index.")
        self.memory[ix] = self.memory[self.length - 1]
        self.memory[self.length - 1] = b""
        self.length -= 1
        self.loto.release()

    def close(self):
        for block in (self.loto.lock, self.memory.shm):
            block.unlink()
            block.close()
        atexit.unregister(self.close)


class AbstractNotepad(ABC):
    def __init__(
        self,
        prefix=None,
        create=False,
        cleanup_on_exit=True,
        index_type = None,
        codec_factory: Optional[Callable] = None,
        mnemonic_factory: Optional[Callable] = None,
        **index_kwargs
    ):
        if prefix is None:
            prefix = randint(100000, 999999)
        self.prefix = prefix
        if mnemonic_factory is not None:
            self.memorize, self.remember = mnemonic_factory()
        if codec_factory is not None:
            if mnemonic_factory is None:
                raise TypeError("can't specify codec without a mnemonic")
            encode, decode = codec_factory()
            self.memorize = partial(self.memorize, encode=encode)
            self.remember = partial(self.remember, decode=decode)
        if index_type is None:
            raise TypeError("must specify an index type")
        try:
            self.index = index_type(
                f"{prefix}_index", create, cleanup_on_exit, **index_kwargs
            )
            self.update_index()
        except FileNotFoundError:
            raise FileNotFoundError(
                "the index space for this object has not been initialized "
                "(or has been deleted) and create=False was passed. Try "
                "constructing this object with create=True."
            )
        atexit.register(self.close)

    def get_raw(self, key):
        return fetch_block_bytes(self.address(key))

    def address(self, key):
        return f"{self.prefix}_{key}"

    def update_index(self, sync=True) -> Union[dict, tuple]:
        return self.index.update(sync)

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def get(self, key, default, fetch):
        pass

    @abstractmethod
    def __setitem__(self, key, value, exists_ok: bool = True):
        pass

    @abstractmethod
    def set(self, key, value, exists_ok: bool = True):
        pass

    def keys(self):
        return self.update_index()

    @abstractmethod
    def values(self):
        pass

    @abstractmethod
    def items(self):
        pass

    def iterkeys(self):
        for key in self.update_index():
            yield key

    def itervalues(self):
        """return an iterator over entries in the cache"""
        for key in self.update_index():
            yield self[key]

    def iteritems(self):
        """return an iterator over key / value pairs in the cache"""
        for key in self.update_index():
            yield key, self[key]

    @abstractmethod
    def __delitem__(self, key):
        pass

    @abstractmethod
    def dump(self, key, fn):
        pass

    @abstractmethod
    def close(self, dump):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()


class Notepad(AbstractNotepad):
    """generic read-write cache"""
    def __init__(
        self,
        prefix=None,
        create=False,
        cleanup_on_exit=True,
        index_type = ListIndex,
        codec_factory=json_pickle_codec_factory,
        mnemonic_factory=generic_mnemonic_factory,
        **index_kwargs
    ):
        super().__init__(
            prefix,
            create,
            cleanup_on_exit,
            index_type,
            codec_factory,
            mnemonic_factory,
            **index_kwargs
        )

    def __getitem__(self, key, fetch=True):
        result = self.remember(self.address(key), fetch)
        if result is None:
            raise KeyError
        return result

    def get(self, key, default=None, fetch=True):
        try:
            return self.__getitem__(key, fetch)
        except KeyError:
            return default

    def __setitem__(self, key, value, exists_ok: bool = True):
        if key in (["index", "index_lock"]):
            raise KeyError("'index' and 'index_lock' are reserved key names")
        try:
            self.memorize(value, self.address(key), exists_ok)
        except FileExistsError:
            raise KeyError(
                f"{key} already exists in this object's cache. pass "
                f"exists_ok=True to overwrite it."
            )
        if key not in self.update_index():
            self.index.add(key)

    def set(self, key: str, value: Any, exists_ok: bool = True):
        return self.__setitem__(key, value, exists_ok)

    def __delitem__(self, key):
        if key in (["index", "index_lock"]):
            raise KeyError("'index' and 'index_lock' are reserved key names")
        try:
            delete_block(self.address(key))
            self.index.remove(key)
        except FileNotFoundError:
            raise KeyError(f"{key} not apparently assigned")

    def values(self):
        # TODO: something with MemoryViews?
        raise NotImplementedError("Try using the itervalues method for now")

    def items(self):
        # TODO: something with MemoryViews?
        raise NotImplementedError("Try using the iteritems method for now")

    def dump(self, key, fn=None, mode="wb"):
        if fn is None:
            fn = f"{self.prefix}_{key}".replace(".", "_")
        with open(fn, mode) as file:
            # noinspection PyTypeChecker
            file.write(self.get_raw(key))

    def close(self, dump=False):
        for key in self.update_index():
            if dump is True:
                self.dump(key)
            del self[key]
        self.index.close()
        atexit.unregister(self.close)

    def clear(self):
        for key in self.update_index():
            del self[key]

    def __str__(self):
        return f"{self.__class__.__name__} with keys {self.update_index()}"

    def __repr__(self):
        return self.__str__()


class GridPaper(AbstractNotepad):

    def __init__(self, prefix=None, create=False, cleanup_on_exit=True):
        # noinspection PyTypeChecker
        super().__init__(
            prefix,
            create,
            cleanup_on_exit,
            index_type=DictIndex,
            mnemonic_factory=numpy_mnemonic_factory
        )
        self._open_blocks = []

    def __getitem__(self, key, fetch=True, copy=True):
        try:
            result, block = self.remember(self.index[key], fetch, copy)
            if copy is False:
                self._open_blocks.append(block)
        except FileNotFoundError:
            raise KeyError(f"{key} not found in this object's space.")
        return result

    def get(self, key, default=None, fetch=True, copy=False):
        try:
            return self.__getitem__(key, fetch, copy)
        except KeyError:
            return default

    def __setitem__(self, key, value, exists_ok: bool = True):
        if key in (["index", "index_lock"]):
            raise KeyError("'index' and 'index_lock' are reserved key names")
        try:
            metadata = self.memorize(value, self.address(key), exists_ok)
        except FileExistsError:
            raise KeyError(
                f"{key} already exists in this object's cache. pass "
                f"exists_ok=True to overwrite it."
            )
        if key not in self.update_index():
            self.index[key] = metadata

    def _add_index_key(self, key):
        raise TypeError

    def set(self, key, value, exists_ok: bool = True):
        return self.__setitem__(key, value, exists_ok)

    def values(self):
        # TODO: something with MemoryViews?
        raise NotImplementedError("Try using the itervalues method for now")

    def items(self):
        # TODO: something with MemoryViews?
        raise NotImplementedError("Try using the iteritems method for now")

    def dump(self, key, fn=None):
        import numpy as np

        if fn is None:
            fn = f"{self.prefix}_{key}".replace(".", "_")
        self[key].tofile(fn)

    def __delitem__(self, key):
        if key in (["index", "index_lock"]):
            raise KeyError("'index' and 'index_lock' are reserved key names")
        try:
            delete_block(self.address(key))
            self.index.remove(key)
            for block in self._open_blocks:
                if block.name == key:
                    block.close()
                    block.unlink()
        except FileNotFoundError:
            raise KeyError(f"{key} not apparently assigned")

    def close(self, dump=False):
        for key in self.update_index():
            if dump is True:
                self.dump(key)
            del self[key]
        self.index.close()
        atexit.unregister(self.close)

    def clear(self):
        for key in self.update_index().keys():
            del self[key]

    def __str__(self):
        return (
            f"{self.__class__.__name__} with keys {self.update_index().keys()}"
        )

    def __repr__(self):
        return self.__str__()


class Sticky:
    def __init__(
        self,
        address=None,
        readonly=True,
        value=None,
        cleanup_on_exit=True
    ):
        self.address = str(address)
        self.encode, self.decode = json_pickle_codec_factory()
        self._cached_value = value
        self.readonly = readonly
        if cleanup_on_exit is True:
            atexit.register(self.close)

    @classmethod
    def note(
        cls,
        obj,
        address=None,
        exists_ok=False,
        readonly=True,
        cleanup_on_exit=True
    ):
        init_kwargs = {
            'value': obj,
            'readonly': readonly,
            'address': str(address),
            'cleanup_on_exit': cleanup_on_exit
        }
        if address is None:
            init_kwargs['address'] = f"{randint(100000, 999999)}"
        note = Sticky(**init_kwargs)
        note.stick(obj)
        return note

    def read(self, reread=False):
        if (
                (self._cached_value is not None)
                and (reread is False)
        ):
            return self._cached_value
        try:
            block = SharedMemory(self.address)
        except FileNotFoundError:
            return None
        stream = block.buf.tobytes()
        self._cached_value = self.decode(stream)
        return self._cached_value

    def close(self):
        block = SharedMemory(self.address)
        block.unlink()
        block.close()

    def pull(self):
        return self.close()

    def stick(self, obj):
        encoded = self.encode(obj)
        size = len(encoded)
        block = create_block(self.address, size)
        block.buf[:] = encoded
        self._cached_value = obj

    def __str__(self):
        if self._cached_value is not None:
            return self._cached_value.__str__()
        else:
            return f'unread sticky'

    def __repr__(self):
        if self._cached_value is not None:
            return f"sticky: {self._cached_value.__repr__()}"
        else:
            return 'unread sticky'


"""
possible TODOs:
1. we could have the index stored at some other randomized address, but
i very much like the idea of these classes being portable between processes
by referencing only a single string

2: dumb version of Notepad without an index for faster assignment. or does
Sticky fully accomplish this?

3. Version of DictIndex that has its own ListIndex for keys, and stores values
in individual shared memory locations for faster assignment 
"""