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
    try:
        block = SharedMemory(address)
    except FileNotFoundError:
        return None
    return block.buf.tobytes()
