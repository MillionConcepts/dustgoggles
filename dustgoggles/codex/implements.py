"""
high-level abstractions for working with Python shared memory.

notes
----

1. on the cleanup_on_exit kwarg:
If SharedMemory objects are instantiated in a top-level process,
especially in the __main__ namespace, the multiprocessing.resource_tracker
process generally does a good job of cleaning them up. However, sometimes we
need a little more manual control for the following reasons:
1) blocks initially created in child processes may not always be cleanly and
automatically unlinked, particularly if the parent process is very
long-running
2) on Mac and Linux, multiprocessing.resource_tracker often does _too_
enthusiastic a job of unlinking SharedMemory objects, making them unstable
when used by processes that do not share a parent, so you might want to
deactivate it: see deactivate_shared_memory_resource_tracker() in
dustgoggles.codex.memutilz. If you do that, but you don't _want_ to create a
memory leak -- which you might, if you are trying to make a persistent
in-memory object cache, but in most cases you probably don't -- someone has
to clean these objects up manually. cleanup_on_exit is a convenient way to
do that. but bear in mind that it still won't effectively clean up in the
event of some kinds of hard crash of the process!
"""

from abc import abstractmethod, ABC
import atexit
from functools import partial
from multiprocessing.shared_memory import ShareableList, SharedMemory
from random import randint, randbytes
import time
from typing import Any, Union, Optional, Callable

from dustgoggles.codex.codecs import (
    generic_mnemonic_factory,
    json_codec_factory,
    json_pickle_codec_factory,
    numpy_mnemonic_factory,
)
from dustgoggles.codex.memutilz import (
    open_block, fetch_block_bytes, delete_block, exists_block
)


class LockoutTagout:
    """
    shared memory synchronization primitive.
    """

    def __init__(self, name=None, create=True):
        if name is None:
            name = randint(100000, 999999)
        self.name = name
        self.tag = randbytes(5)
        self.lock = open_block(f"{self.name}_lock", 5, create)

    def acquire(self, timeout=0.1, increment=0.0001):
        """
        try to acquire a lock (which is shared by other LockoutTagout
        objects in this and other processes with the same name). unless the
        "lock" contains this object's personal random "tag" (meaning that the
        lock has already been acquired by this object), block until the space
        is free, checking every `increment` seconds, until `timeout` is
        reached. setting `timeout` to a negative number disables timeout.
        """
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
                if (timeout >= 0) and (time_waiting > timeout):
                    raise TimeoutError("timed out waiting for lock")
                time.sleep(increment)

    def release(self, release_only_mine=True):
        tag = self.lock.buf.tobytes()
        if (tag != self.tag) and (release_only_mine is True):
            raise ConnectionRefusedError
        self.lock.buf[:] = b"\x00\x00\x00\x00\x00"

    def close(self):
        delete_block(self.lock.name)



class FakeLockoutTagout:
    """
    shared memory synchronization that doesn't synchronize or use shared
    memory. can be used as a mock object for tests, or passed to
    ShareableIndex objects to decrease their thread safety in exchange for
    modest boosts to performance.
    """

    def acquire(self, *args, **kwargs):
        pass

    def release(self, *args, **kwargs):
        pass


class ShareableIndex(ABC):
    """
    abstract base class for indexes that exist in shared memory and can be
    accessed from multiple processes. they are used to back various sorts of
    notepads in this module, but can also be used independently.
    """

    def __init__(
        self,
        name: str = None,
        create: bool = True,
        cleanup_on_exit: bool = False,
        no_lockout: bool = False,
        **_
    ):
        if (create is False) and not exists_block(name):
            raise FileNotFoundError(
                f"Shared memory block {name} does not exist and "
                f"create=False passed. Construct this index with "
                f"create=True if you want to initialize a new index."
            )
        if no_lockout is True:
            self.loto = FakeLockoutTagout()
        else:
            self.loto = LockoutTagout(name)
        if cleanup_on_exit is True:
            atexit.register(self.close)

    @abstractmethod
    def update(self, sync: bool = True):
        """
        retrieve the contents of this index from shared memory. if you're
        sure they haven't changed, sync=False only retrieves the index if
        there are no locally-cached values, which is much faster but does not
        guarantee thread safety.
        """
        pass

    @abstractmethod
    def add(self, key, value=None):
        """
        add a key to the index. Some types of index support assigning a value
        to individual keys; some do not.
        """
        pass

    @abstractmethod
    def __delitem__(self, key):
        """delete an item from the index."""
        pass

    @abstractmethod
    def close(self):
        """destroy this object and everything about it."""
        pass

    def remove(self, key):
        """"remove a key from this index."""
        return self.__delitem__(key)


class DictIndex(ShareableIndex):
    """
    very simple index based on dumping json into a shared memory block.

    note that object keys must be strings. integers, floats, etc. used as
    keys will be converted to strings by json serialization / deserialization.
    """
    def __init__(
        self,
        name: str=None,
        create: bool=True,
        cleanup_on_exit: bool=False,
        no_lockout: bool = False
    ):
        """
        Args:
            name: name / address of the shared memory backing the index.
            create: create a new index if none exists. will _not_ overwrite an
                existing index.
            cleanup_on_exit: delete everything about this object on process
                exit. see notes on cleanup_on_exit in the top-level docstring
                for codex.implements.
            no_lockout: don't use a lock on this index. decreases thread
                safety for modest gains in performance.
        """
        super().__init__(name, create, cleanup_on_exit)
        memorize, remember = generic_mnemonic_factory()
        encode, decode = json_codec_factory()
        self.name = name
        self.memorize = partial(
            memorize, address=self.name, exists_ok=True, encode=encode
        )
        self.remember = partial(
            remember, address=self.name, fetch=True, decode=decode
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

    def add(self, key, value=None):
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
        create=True,
        cleanup_on_exit=False,
        no_lockout=False,
        max_length=64,
        max_characters=64,
    ):
        """
        Args:
            name: name / address of the shared memory backing the index.
            create: create a new index if none exists. will _not_
                overwrite an existing index.
            cleanup_on_exit: delete everything about this object on
                process  exit. see notes on cleanup_on_exit in the top-level
                docstring for codex.implements.
            no_lockout: don't use a lock on this index. decreases thread
                safety for modest gains in performance.
            max_length: maximum length of index. a shorter index is faster.
            max_characters: maximum number of characters per index entry.
                fewer is faster.
        """
        super().__init__(name, create, cleanup_on_exit, no_lockout)
        if create is False:
            self.memory = ShareableList(name)
        else:
            self.memory = ShareableList(
                [b"\x00" * max_characters for _ in range(max_length)],
                name=name
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
        cleanup_on_exit=False,
        index_type=None,
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
        if cleanup_on_exit is True:
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
        cleanup_on_exit=False,
        index_type=ListIndex,
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

    def __init__(self, prefix=None, create=False, cleanup_on_exit=False):
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

    def __setitem__(
        self, key, value, exists_ok: bool = True, keep_open: bool = False
    ):
        if key in (["index", "index_lock"]):
            raise KeyError("'index' and 'index_lock' are reserved key names")
        try:
            metadata, block = self.memorize(
                value, self.address(key), exists_ok, keep_open
            )
            if keep_open is True:
                self._open_blocks.append(block)
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
        # readonly=True,
        value=None,
        cleanup_on_exit=False
    ):
        self.address = str(address)
        self.encode, self.decode = json_pickle_codec_factory()
        self._cached_value = value
        # self.readonly = readonly
        if cleanup_on_exit is True:
            atexit.register(self.close)

    @classmethod
    def note(
        cls,
        obj,
        address=None,
        # readonly=True,
        cleanup_on_exit=False
    ):
        init_kwargs = {
            'value': obj,
            # 'readonly': readonly,
            'address': str(address),
            'cleanup_on_exit': cleanup_on_exit
        }
        if address is None:
            init_kwargs['address'] = f"{randint(100000, 999999)}"
        note = Sticky(**init_kwargs)
        note.stick(obj)
        return note

    def read(self, reread=False):
        if (self._cached_value is not None) and (reread is False):
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
        block = open_block(self.address, size)
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
