import sys
from multiprocessing.shared_memory import SharedMemory

def create_block(address, size, exists_ok=True):
    try:
        return SharedMemory(address, size=size, create=True)
    except FileExistsError:
        if exists_ok is False:
            raise
        old_block = SharedMemory(address)
        old_block.unlink()
        old_block.close()
        return SharedMemory(address, create=True, size=size)


def fetch_block_bytes(address):
    block = SharedMemory(address)
    return block.buf.tobytes()


def exists_block(address):
    try:
        SharedMemory(address)
        return True
    except FileNotFoundError:
        return False


def delete_block(address):
    block = SharedMemory(address)
    block.unlink()
    block.close()


def monkeypatch_resource_tracker():
    """
    monkeypatch resource tracker on posix systems to handle longstanding
    excessively-enthusiastic shared-memory gc issue.

    note that this _will_ permit memory leaks from non-gracefully-terminated
    processes if you call it with excessive enthusiasm and aren't tracking
    the shared memory blocks anywhere else.

    see: https://github.com/python/cpython/issues/82300 (also direct source
    for this code)
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


