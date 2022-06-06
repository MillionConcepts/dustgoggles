"""
high-level abstractions for working with Python shared memory.

notes
----
## indexes and locks: performance
Like any synchronization objects, shared indices and locks introduce some
overhead. The overhead is negligible for applications that access these
objects somewhat infrequently, but can be significant for applications that
access them at high rates, especially from many threads, as processes spinlock
waiting for access to shared resources.

If you are certain that your operations are intrinsically thread-safe (perhaps
processes have some other synchronization strategy), or your application is
tolerant of failed or corrupted writes, you can deactivate locking by passing
no_lockout=True to AbstractNotepad or ShareableIndex constructors. (This will
propagate from an AbstractNotepad to its index.)

For even more performance improvements, you can pass index_type=FakeIndex to
a Notepad constructor. This will, of course, completely deactivate indexing:
you will not be able to explicitly list the keys of the TagIndex. This may be
acceptable if you have top-down control over all processes sharing this
Notepad address space or are otherwise able to a priori determine the possible
set of keys.

This WILL NOT work with GridPaper, which requires metadata stored in a
MetaTagIndex in order to correctly format the numpy arrays it retrieves from
shared memory.

The use of FakeIndex also makes automated cleanup with .close() or
cleanup_on_exit impossible, and you must either rely on the resource tracker,
explicitly unlink any leftover shared memory blocks on exit, or accept the
possibility of memory leaks.

## cleanup_on_exit / deactivate_shared_memory_resource_tracker()
If SharedMemory objects are instantiated in a top-level process,
especially in the __main__ namespace, the multiprocessing.resource_tracker
process generally does a good job of cleaning them up. However, sometimes we
need a little more manual control for the following reasons:
    1. blocks initially created in child processes may not always be cleanly
    and automatically unlinked, particularly if:
        a. they are killed at the OS level
        b. they never terminate
        c. the parent process is very long-running
        d. they raise especially gnarly exceptions
    2. on Mac and Linux, multiprocessing.resource_tracker often does too
    good a job of unlinking SharedMemory objects, making them unstable
    under the following conditions:
        a. use by processes that do not inherit a resource tracker from the
        same parent -- this can include both processes executed by separate
        parents and processes executed by parents that did not themselves
        instantiate a resource tracker for their children to inherit
        b. maybe others
    3. also on Mac and Linux, high-rate updates (> ~40/sec) to memory blocks
    can confuse the resource tracker, causing it to throw spurious warnings
    about leaked shared memory objects if everything is in fact unlinked
    cleanly -- these are usually followed by non-spurious but pointless
    warnings that it can't unlink those already-unlinked blocks. These
    warnings occur at exit and so are difficult to suppress.
For these reasons, you might simply want to deactivate the resource tracker:
see deactivate_shared_memory_resource_tracker() in dustgoggles.codex.memutilz.
If you do that, but you don't _want_ to create a memory leak -- which you
might, if you are trying to make a persistent in-memory object cache, but in
most cases you probably don't -- someone has to clean these objects up
manually.

Cleanup_on_exit is a convenient way to handle either this cleanup issue or
listed in 1.a - 1.d, but bear in mind that it still won't effectively clean up
in the event of some kinds of hard crash of the topmost process! You also
probably don't want to set it inside processes that will terminate before you
need to use shared memory objects they created again.

If you _do_ experience memory leaks due to unavoidable instability in the non-
dustgoggles portions of your code, they are fairly easy to clean up in most
Mac / Linux Python implementations, as SharedMemory objects are simply files
in /dev/shm and can be freely deleted at the OS level using rm or whatever. It
is somewhat more challenging to free leaked blocks on Windows.
"""
import atexit
import datetime as dt
import os
import time
from abc import abstractmethod, ABC
from collections import namedtuple
from functools import partial
from hashlib import sha256
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from random import randint
from typing import Any, Union, Optional, Callable

from cytoolz import juxt

from dustgoggles.codex.codecs import (
    generic_mnemonic_factory,
    json_codec_factory,
    pickle_codec_factory,
    numpy_mnemonic_factory,
)
from dustgoggles.codex.memutilz import (
    open_block, fetch_block_bytes, delete_block, exists_block
)
from dustgoggles.func import zero


# debug helpers

def here():
    return f"{dt.datetime.now().isoformat()[-9:]},{os.getpid()}"


def log(message):
    with open("../holding/dumplog.csv", "a+") as file:
        file.write(message + "\n")


def printstack(stack):
    return ";".join(
        [
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.function}"
            for frame in stack
        ]
    )


HASHER = partial(sha256, usedforsecurity=False)


class FakeLock:
    """
    shared memory synchronization that doesn't synchronize or use shared
    memory. can be used as a mock object for tests, or passed to
    ShareableIndex objects to decrease their thread safety in exchange for
    modest boosts to performance.
    """
    def __init__(self, *_, **__):
        FakeLockTuple = namedtuple("FakeLock", ["unlink", "close"])
        self.lock = FakeLockTuple(unlink=zero, close=zero)
        self.debug = False

    def __enter__(self):
        pass

    def __exit__(self, *_, **__):
        pass

    def acquire(self, *args, **kwargs):
        pass

    def release(self, *args, **kwargs):
        pass

class SlidingLock:
    """
    shared memory synchronization primitive. uses shared memory 'filesystem'
    backend as an atomic write register.
    """
    def __init__(self, name, debug=False):
        self.name = name
        self.lock = None
        self.count = open_block(f"{self.name}_count", 5)
        self.number = 0
        self.floor = None
        self.marker = None
        self.acquired = False
        self.spins = 0
        self.max_spins = None
        self.delay = 0
        self.debug = debug

    def __enter__(self):
        try:
            self._renumber()
            self.acquire()
        except Exception as ex:
            if getattr(self, "debug") is True:
                log(f"{here()},{self.number},crashing out: {ex}")
            self.release()
            raise ex
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if getattr(self, "debug") is True:
            log(f"{here()},{self.number},entering cleanup")
        self.release()

    def _renumber(self):
        self.floor = int.from_bytes(self.count.buf.tobytes(), "little")
        self.number = self.floor + 1
        while self.marker is None:
            try:
                self.marker = SharedMemory(
                    f"{self.name}_{self.number}", size=1, create=True
                )
            except FileExistsError:
                self.number += 1
        if getattr(self, "debug") is True:
            log(f"{here()},{self.number},taking slot {self.number}")

    def _check_acquirement(self):
        for other_number in reversed(range(self.floor, self.number)):
            if exists_block(f"{self.name}_{other_number}_lock"):
                if getattr(self, "debug") is True:
                    log(f"{here()},{self.number},{other_number} has precedence")
                # # # # #
                self.wait()
                return False
        if getattr(self, "debug") is True:
            log(f"{here()},{self.number},taking precedence")
        return True

    def acquire(self):
        try:
            self.lock = SharedMemory(
                f"{self.name}_{self.number}_lock", size=1, create=True
            )
            if getattr(self, "debug") is True:
                log(f"{here()},{self.number},placed lock")
        except FileExistsError:
            if getattr(self, "debug") is True:
                log(f"{here()},{self.number},already have lock")
            self.acquired = True
        while self.acquired is False:
            self.acquired = self._check_acquirement()

    def wait(self):
        self.spins += 1
        if self.max_spins is not None:
            if self.spins > self.max_spins:
                if getattr(self, "debug") is True:
                    log(f"{here()},{self.number},failure (waiting timeout)")
                raise TimeoutError("timed out waiting for lock")
        time.sleep(self.delay)

    def release(self):
        if self.acquired is True:
            self.count.buf[:] = self.number.to_bytes(5, "little")
            if getattr(self, "debug") is True:
                log(f"{here()},{self.number},set floor to {self.number}")
        for thing in ("lock", "marker"):
            block = getattr(self, thing)
            if block is None:
                continue
            block.unlink()
            block.close()
            setattr(self, thing, None)
            if getattr(self, "debug") is True:
                log(f"{here()},{self.number},released {thing}")
        self.acquired = False

    def close(self):
        for obj in self.lock, self.count, self.marker:
            try:
                delete_block(obj.name)
            except (FileNotFoundError, TypeError, AttributeError):
                continue


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
        debug=False,
        **_
    ):
        if (create is False) and not exists_block(name):
            raise FileNotFoundError(
                f"Shared memory block {name} does not exist and "
                f"create=False passed. Construct this index with "
                f"create=True if you want to initialize a new index."
            )
        if no_lockout is True:
            self.lock_class = FakeLock
        else:
            self.lock_class = SlidingLock
        if cleanup_on_exit is True:
            atexit.register(self.close)
        self.debug = debug

    @abstractmethod
    def update(self):
        """
        retrieve the contents of this index from shared memory.
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


class FakeIndex(ShareableIndex):
    def __init__(self, *_, **__):
        super().__init__(no_lockout=True)

    def update(self, *_, **__):
        pass

    def add(self, *_, **__):
        pass

    def __delitem__(self, *_, **__):
        pass

    def close(self, *_, **__):
        pass

    def remove(self, *_, **__):
        pass


class TagIndex(ShareableIndex):
    """
    fast index based around predictable link structures and reverse hash maps
    in shared memory
    """
    def __init__(
        self,
        name=None,
        create=True,
        cleanup_on_exit=False,
        debug=False,
        no_lockout=False,
    ):
        """
        Args:
            name: name / address of the shared memory backing the index.
            create: create a new index if none exists. will _not_
                overwrite an existing index.
            cleanup_on_exit: delete everything about this object on
                process  exit. see notes on cleanup_on_exit in the top-level
                docstring for codex.implements.
            debug: dump debug information to a log file.
            no_lockout: don't use a lock on this index. decreases thread
                safety for modest gains in performance.
        """
        if name is None:
            name = str(randint(100000, 999999))
        self.name = name
        super().__init__(name, create, cleanup_on_exit, no_lockout, debug)
        if create is False:
            self.memory = SharedMemory(name=name)
        else:
            self.memory = SharedMemory(name=name, create=True, size=8)
        self.cache = []
        self.length = None

    def hash_address(self, key):
        if not isinstance(key, bytes):
            key = str(key).encode()
        namehash = HASHER(key).hexdigest()
        return f"{self.name}_name_{namehash}"

    def key_address(self, index_ix):
        return f"{self.name}_key_{index_ix}"

    def update_length(self):
        try:
            self.length = int.from_bytes(self.memory.buf.tobytes(), "little")
        except ValueError as error:
            if "not enough values" not in str(error):
                raise error
            return self.update_length()

    def update(self):
        self.cache = []
        for ix, key in self._enumerate_keys():
            self.cache.append(key)
        return self.cache

    def check(self, target) -> Optional[int]:
        if exists_block(self.hash_address(target)):
            nameblock = SharedMemory(self.hash_address(target))
            return int.from_bytes(nameblock.buf.tobytes(), "little")
        return None

    def _enumerate_keys(self):
        self.update_length()
        for ix in range(self.length):
            try:
                keyblock = SharedMemory(self.key_address(ix))
                key = keyblock.buf.tobytes().decode()
            except FileNotFoundError:
                continue
            yield ix, key

    def _add(self, key, exists_ok) -> Optional[int]:
        found_at = self.check(key)
        if found_at is not None:
            if exists_ok is True:
                if getattr(self, "debug") is True:
                    log(f"{here()},,{key} already in index at {found_at}")
                return found_at
            raise FileExistsError
        self.update_length()
        new_length_bytes = (self.length + 1).to_bytes(8, "little")
        self.memory.buf[:] = new_length_bytes
        keybytes = str(key).encode()
        keyblock = open_block(
            self.key_address(self.length),
            create=True,
            size=len(keybytes),
            overwrite=True
        )
        keyblock.buf[:] = keybytes
        position_bytes = self.length.to_bytes(8, "little")
        nameblock = open_block(self.hash_address(key), create=True, size=8)
        nameblock.buf[:] = position_bytes
        if getattr(self, "debug") is True:
            log(f"{here()},,set {self.length} to {key}")
        return self.length

    def add(self, key, value=None, exists_ok=True):
        if value is not None:
            raise ValueError("TagIndex does not support value-setting.")
        with self.lock_class(self.name, self.debug):
            return self._add(key, exists_ok)

    def _delitem(self, target):
        found_at = self.check(target)
        if found_at is None:
            raise KeyError(f"{target} not found in this index.")
        self.update_length()
        if found_at != self.length - 1:
            lastblock = SharedMemory(self.key_address(self.length - 1))
            lastkey = lastblock.buf.tobytes()
            moved_block = open_block(
                self.key_address(found_at),
                size=len(lastkey),
                overwrite=True
            )
            moved_block.buf[:] = lastkey
            nameblock = SharedMemory(self.hash_address(lastkey))
            nameblock.buf[:] = found_at.to_bytes(8, "little")
            lastblock.unlink()
            lastblock.close()
            if getattr(self, "debug") is True:
                log(
                    f"{here()},,overwrote block at position {found_at} "
                    f"with block at position {self.length - 1} "
                )
        else:
            delete_block(self.key_address(found_at))
            if getattr(self, "debug") is True:
                log(f"{here()},,deleted block at position {found_at}")
        delete_block(self.hash_address(target))
        new_length_bytes = (self.length - 1).to_bytes(8, "little")
        self.memory.buf[:] = new_length_bytes
        if getattr(self, "debug") is True:
            log(
                f"{here()},,deleted {target} at index slot {found_at} "
                f"and decremented index counter"
            )
        return found_at

    def __delitem__(self, target):
        with self.lock_class(self.name, self.debug):
            return self._delitem(target)

    def close(self):
        self.memory.unlink()
        self.memory.close()
        if exists_block(f"{self.name}_count"):
            block = SharedMemory(f"{self.name}_count")
            block.unlink()
            block.close()
        atexit.unregister(self.close)


class MetaTagIndex(TagIndex):
    """
    very slightly slower and slightly more complex version of TagIndex that
    permits attaching metadata to keys. reference usage is in GridPaper, which
    uses a MetaTagIndex to store metadata about dtypes and dimensions of numpy
    arrays in order to accurately and rapidly recreate them from raw bytes.
    """
    def __init__(
        self,
        name=None,
        create=True,
        cleanup_on_exit=False,
        debug=False,
        no_lockout=False,
    ):
        """
        Args:
            name: name / address of the shared memory backing the index.
            create: create a new index if none exists. will _not_ overwrite an
                existing index.
            cleanup_on_exit: delete everything about this object on process
                exit. see notes on cleanup_on_exit in the top-level docstring
                for codex.implements.
            debug: activate debug-level logging
            no_lockout: don't use a lock on this index. decreases thread
                safety for modest gains in performance.
        """
        super().__init__(name, create, cleanup_on_exit, debug, no_lockout)
        memorize, remember = generic_mnemonic_factory()
        encode, decode = json_codec_factory()
        self.memorize = partial(memorize, exists_ok=True, encode=encode)
        self.remember = partial(remember, fetch=True, decode=decode)
        if create is True:
            self.memory = SharedMemory(self.name)

    def val_address(self, index_ix):
        return f"{self.name}_val_{index_ix}"

    def _load_key(self, ix):
        keyblock = SharedMemory(self.key_address(ix))
        return keyblock.buf.tobytes().decode()

    def _load_value(self, ix):
        try:
            return self.remember(self.val_address(ix))
        except FileNotFoundError:
            return

    def _enumerate(self, get_keys=False, get_values=False):
        loaders = [
            loader if arg is True else zero
            for arg, loader
            in zip((get_keys, get_values), (self._load_key, self._load_value))
        ]
        load = juxt(loaders)
        self.update_length()
        for ix in range(self.length):
            key, val = load(ix)
            output = [ix]
            if get_keys is True:
                output.append(key)
            if get_values is True:
                output.append(val)
            yield tuple(output)

    def keys(self):
        return [key for _, key in self._enumerate(True)]

    def values(self):
        return [val for _, val in self._enumerate(False, True)]

    def items(self):
        return [(key, val) for _, key, val in self._enumerate(True, True)]

    def add(self, key, value=None, exists_ok=True):
        with self.lock_class(self.name, self.debug):
            writehead = super()._add(key, exists_ok)
            if value is not None:
                self.memorize(value, self.val_address(writehead))
            return writehead

    def __getitem__(self, key):
        found_at = self.check(key)
        if found_at is None:
            raise KeyError(f"{key} not found in index.")
        valblock = SharedMemory(self.val_address(found_at))
        return valblock.buf.tobytes()

    def __delitem__(self, target):
        with self.lock_class(self.name, self.debug):
            found_at = super().__delitem__(target)
            if (
                (found_at == self.length - 1)
                or not exists_block(self.val_address(self.length - 1))
            ):
                delete_block(self.val_address(found_at), True)
            elif exists_block(self.val_address(self.length - 1)):
                valblock = SharedMemory(self.val_address(self.length - 1))
                valbytes = valblock.buf.tobytes()
                moved_block = open_block(
                    self.val_address(found_at), size=len(valbytes), overwrite=True
                )
                moved_block.buf[:] = valbytes
                delete_block(self.val_address(self.length - 1))
        return found_at


class AbstractNotepad(ABC):
    def __init__(
        self,
        prefix=None,
        create=False,
        cleanup_on_exit=False,
        index_type=None,
        codec_factory: Optional[Callable] = None,
        mnemonic_factory: Optional[Callable] = None,
        update_on_init=True,
        debug=False,
        **index_kwargs
    ):
        self.debug = debug
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
                f"{prefix}_index",
                create,
                cleanup_on_exit,
                debug,
                **index_kwargs
            )
            if update_on_init is True:
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

    def update_index(self) -> Union[dict, tuple]:
        return self.index.update()

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
        index_type=TagIndex,
        codec_factory=pickle_codec_factory,
        mnemonic_factory=generic_mnemonic_factory,
        update_on_init=False,
        debug=False,
        **index_kwargs
    ):
        super().__init__(
            prefix,
            create,
            cleanup_on_exit,
            index_type,
            codec_factory,
            mnemonic_factory,
            update_on_init,
            debug,
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

    def __setitem__(self, key, value, meta=None, exists_ok: bool = True):
        if isinstance(key, str) and key.startswith("index"):
            raise KeyError("'index' is a reserved key prefix")
        try:
            self.index.add(key, meta, exists_ok=exists_ok)
            self.memorize(value, self.address(key), exists_ok)
            if self.debug is True:
                log(f"{here()},,set key={key} to {value}")
        except FileExistsError:
            raise KeyError(
                f"{key} already exists in this object's cache. pass "
                f"exists_ok=True to overwrite it."
            )

    def set(self, key: str, value: Any, meta=None, exists_ok: bool = True):
        return self.__setitem__(key, value, meta, exists_ok)

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
        try:
            keys = self.update_index().copy()
            for key in keys:
                if dump is True:
                    self.dump(key)
                del self[key]
            self.index.close()
        except AttributeError:
            pass
        atexit.unregister(self.close)

    def clear(self):
        for key in self.update_index():
            del self[key]

    def __str__(self):
        return f"{self.__class__.__name__} with keys {self.update_index()}"

    def __repr__(self):
        return self.__str__()


class GridPaper(AbstractNotepad):

    def __init__(
        self,
        prefix=None,
        create=False,
        cleanup_on_exit=False,
        index_type=MetaTagIndex,
        update_on_init=False,
        debug=False,
        **index_kwargs
    ):
        # noinspection PyTypeChecker
        super().__init__(
            prefix,
            create,
            cleanup_on_exit,
            index_type,
            mnemonic_factory=numpy_mnemonic_factory,
            update_on_init=update_on_init,
            debug=debug,
            **index_kwargs
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
        if isinstance(key, str) and key.startswith("index"):
            raise KeyError("'index' is a reserved namespace.")
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
        self.index.add(key, metadata)

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
            f"{self.__class__.__name__} with keys {self.index.keys()}"
        )

    def __repr__(self):
        return self.__str__()


class Sticky:
    def __init__(
        self,
        address=None,
        value=None,
        cleanup_on_exit=False
    ):
        self.address = str(address)
        self.encode, self.decode = pickle_codec_factory()
        self._cached_value = value
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
