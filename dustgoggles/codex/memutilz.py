import sys
from multiprocessing.shared_memory import SharedMemory


def create_block(address, size, exists_ok=True):
    """
    create and return a shared memory block with the specified size,
    unlinking any existing block at that address if exists_ok is not False.
    """
    try:
        return SharedMemory(address, size=size, create=True)
    except FileExistsError:
        if exists_ok is False:
            raise
        old_block = SharedMemory(address)
        old_block.unlink()
        old_block.close()
        return SharedMemory(address, create=True, size=size)


def fetch_block_bytes(address: str) -> bytes:
    """
    get the contents of the shared memory block at address as a binary blob.
    """
    block = SharedMemory(address)
    return block.buf.tobytes()


def exists_block(address: str) -> bool:
    """
    return True if a shared memory block exists at address; False if not.
    """
    try:
        SharedMemory(address)
        return True
    except FileNotFoundError:
        return False


def delete_block(address: str):
    """delete the shared memory block at address."""
    block = SharedMemory(address)
    block.unlink()
    block.close()


def deactivate_shared_memory_resource_tracker():
    """
    monkeypatch resource tracker on posix systems to handle longstanding
    excessively-enthusiastic shared-memory gc issue.

    see: https://github.com/python/cpython/issues/82300 (also direct source
    for this code)

    note that this _will_ permit memory leaks from non-gracefully-terminated
    processes if you call it with excessive enthusiasm and aren't tracking
    the shared memory blocks anywhere else.
    """
    if sys.platform in ('win32', 'cygwin'):
        return

    # noinspection PyUnresolvedReferences
    from multiprocessing import resource_tracker

    # noinspection PyUnresolvedReferences
    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register

    # noinspection PyUnresolvedReferences
    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


