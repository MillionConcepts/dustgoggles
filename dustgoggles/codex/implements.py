from abc import abstractmethod, ABC
import atexit
import time
from multiprocessing.shared_memory import ShareableList, SharedMemory
from random import randint, randbytes
from typing import Any

from dustgoggles.codex.array_codecs import numpy_encoder, numpy_decoder
from dustgoggles.codex.codecs import (
    json_pickle_encoder,
    json_pickle_decoder,
    remember_generic,
    memorize_generic
)
from dustgoggles.codex.memutilz import create_block, fetch_block_bytes


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
            elif tag == b"\x00\x00\x00\x00":
                self.lock.buf[:] = self.tag
            # the extra loop here is for a possibly-unnecessary double check
            # that someone else didn't write their key to the lock at the
            # exact same time.
            else:
                time_waiting += increment
                if increment > timeout:
                    raise TimeoutError("timed out waiting for lock")
                time.sleep(increment)

    def release(self, release_only_mine=True):
        tag = self.lock.buf.tobytes()
        if (tag != self.tag) and (release_only_mine is True):
            raise ConnectionRefusedError
        self.lock.buf[:] = b"\x00\x00\x00\x00"


class ShareableIndex(ABC):

    def __init__(self, name=None, cleanup_on_exit=True):
        self.loto = LockoutTagout(name)
        if cleanup_on_exit is True:
            atexit.register(self.close)

    @abstractmethod
    def add(self, key, value=None):
        pass

    @abstractmethod
    def __delitem__(self, key):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def close(self):
        pass

    def remove(self, key):
        return self.__delitem__(key)


class DictIndex(LockoutTagout):
    pass


class ShareableListIndex(ShareableIndex):
    """
    limited but very fast index using
    multiprocessing.shared_memory.ShareableList
    """
    def __init__(
        self, name=None, length=256, max_characters=128
    ):
        super().__init__(name)
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
        return self._cache

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



# TODO: we could have the index stored at some other randomized address, but
#  i very much like the idea of these classes being portable between processes
#  by referencing only a single string
class Paper:
    """parent class for notes. defines only index methods."""

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix
        self._index_cache = []
        self._index_length = 0

    def address(self, key):
        return f"{self.prefix}_{key}"

    def _index_memory(self):
        return ShareableList(name=self.address("index"))

    def index(self, sync=True):
        if sync is True:
            # TODO: optimize all this stuff with lookahead and lookbehind
            #  for speed, etc.
            self._index_cache = []
            ix = 0
            for ix, key in enumerate(self._index_memory()):
                if key == b"":
                    break
                self._index_cache.append(key)
            self._index_length = ix
        return self._index_cache

    def __str__(self):
        return f"{self.__class__.__name__} with keys {self.index()}"

    def __repr__(self):
        return self.__str__()


# TODO: dumb version without an index for faster assignment


class NoteViewer(Paper):
    """read-only notepad"""

    def __init__(self, prefix, decoder=json_pickle_decoder):
        super().__init__(prefix)
        self.decoder = decoder

    def get_raw(self, key):
        return fetch_block_bytes(self.address(key))

    # TODO: should I raise errors instead of returning none for missing keys
    #  when accessed with slice notation? that is more 'standard'
    def __getitem__(self, key, decoder=json_pickle_decoder):
        stream = self.get_raw(key)
        if stream is None:
            return stream
        return decoder(stream)

    def get(self, key):
        return self.__getitem__(key)

    def keys(self):
        return self.index()

    def iterkeys(self):
        for key in self.index():
            yield key

    def itervalues(self):
        """return an iterator over entries in the cache"""
        for key in self.index():
            yield self[key]

    def iteritems(self):
        """return an iterator over key / value pairs in the cache"""
        for key in self.index():
            yield key, self[key]

    def dump(self, key, fn=None, mode="wb"):
        if fn is None:
            fn = f"{self.prefix}_{key}".replace(".", "_")
        with open(fn, mode) as file:
            # noinspection PyTypeChecker
            file.write(self.get_raw(key))


class Notepad(NoteViewer):
    """full read-write notepad"""

    def __init__(
            self,
            prefix,
            encoder=json_pickle_encoder,
            decoder=json_pickle_decoder,
            memorizer=memorize_generic,
            recaller=remember_generic
    ):
        super().__init__(prefix, decoder)
        try:
            self.index()
        except FileNotFoundError:
            raise FileNotFoundError(
                "the memory space for this Notepad has not been initialized "
                "(or has been deleted). Try constructing it with "
                "Notepad.open()."
            )
        self.encoder = encoder
        self.decoder = decoder
        self.memorize = memorizer
        self.recall = recaller
        self._lock_key = randbytes(4)

    def __setitem__(self, key: str, value: Any, exists_ok: bool = True):
        if key in (["index", "index_lock"]):
            raise KeyError("'index' and 'index_lock' are reserved key names")
        try:
            self.memorize(self.address(key), value, self.encoder, exists_ok)
        except FileExistsError:
            raise KeyError(
                f"{key} already exists in this object's cache. pass "
                f"exists_ok=True to overwrite it."
            )
        if key not in self.index():
            self._add_index_key(self.address(key))

    def __delitem__(self, key):
        if key in (["index", "index_lock"]):
            raise KeyError("'index' and 'index_lock' are reserved key names")
        try:
            block = SharedMemory(self.address(key))
            block.unlink()
            block.close()
            self._remove_index_key(key)
        except FileNotFoundError:
            raise KeyError(f"{key} not apparently assigned")

    def set(self, key, value):
        return self.__setitem__(key, value)

    def close(self, dump=False):
        for key in self.index():
            if dump is True:
                self.dump(key)
            del self[key]
        for block in self._lock_memory(), self._index_memory().shm:
            block.unlink()
            block.close()
        atexit.unregister(self.close)

    def clear(self):
        for key in self.index():
            del self[key]

    def _add_index_key(self, key):
        self._acquire_index_lock()
        self.index()
        self._index_memory()[self._index_length] = key
        self._release_index_lock()

    def _remove_index_key(self, key):
        self._acquire_index_lock()
        index = self.index()
        index_memory = self._index_memory()
        iterator = enumerate(index)
        ix, item = None, None
        while item != key:
            try:
                ix, item = next(iterator)
            except StopIteration:
                raise KeyError(f"{key} not found in this Notepad's index.")
        index_memory[ix] = index_memory[len(index) - 1]
        index_memory[len(index) - 1] = b""
        self._release_index_lock()

    @classmethod
    def open(
            cls,
            prefix=None,
            index_length=256,
            max_key_characters=128,
            exists_ok=True,
            cleanup_on_exit=True,
            **init_kwargs,
    ):
        if prefix is None:
            prefix = randint(100000, 999999)

        # TODO: handle exists_ok for ShareableList
        _index = ListIndex(f"{prefix}_index", index_length, max_key_characters)
        notepad = Notepad(prefix, **init_kwargs)
        if cleanup_on_exit is True:
            # if SharedMemory objects are instantiated in __main__,
            # multiprocessing.util._exit_function() generally does a good job
            # of cleaning them up. however, blocks created in child processes
            # will often not be cleanly and automatically unlinked.
            atexit.register(notepad.close)
        return notepad


class GraphPaper(Notepad):
    def __init__(self, prefix):
        from array_codecs import memorize_array
        super().__init__(
            prefix,
            encoder=numpy_encoder,
            decoder = numpy_decoder,
            memorizer=memorize_array,
            recaller = remember_array
        )


class Sticky:
    def __init__(
            self,
            address=None,
            decoder=json_pickle_decoder,
            readonly=True,
            value=None,
            cleanup_on_exit=True
    ):
        self.address = str(address)
        self.decoder = decoder
        self._cached_value = value
        self.readonly = readonly
        if cleanup_on_exit is True:
            atexit.register(self.close)

    @classmethod
    def note(
            cls,
            obj,
            address=None,
            encoder=json_pickle_encoder,
            decoder=json_pickle_decoder,
            exists_ok=False,
            readonly=True,
            cleanup_on_exit=True
    ):
        init_kwargs = {
            'value': obj,
            'readonly': readonly,
            'decoder': decoder,
            'address': str(address),
            'cleanup_on_exit': cleanup_on_exit
        }
        if address is None:
            init_kwargs['address'] = f"{randint(100000, 999999)}"
        encoded = encoder(obj)
        size = len(encoded)
        block = create_block(init_kwargs['address'], size, exists_ok=exists_ok)
        block.buf[:] = encoded
        return Sticky(**init_kwargs)

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
        self._cached_value = self.decoder(stream)
        return self._cached_value

    def close(self):
        block = SharedMemory(self.address)
        block.unlink()
        block.close()

    def pull(self):
        return self.close()

    def stick(self, obj, encoder=json_pickle_encoder):
        encoded = encoder(obj)
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
