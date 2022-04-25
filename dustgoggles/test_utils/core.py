import random
import string
from multiprocessing import Pipe, Process
from typing import Hashable, Callable

from cytoolz import get_in

from dustgoggles.structures import NestingDict

RNG = random.Random()


# TODO, maybe: make this a hypothesis strategy?
def randval(itemtype):
    if itemtype is str:
        return "".join(RNG.choices(string.ascii_letters, k=RNG.randint(1, 10)))
    elif itemtype is int:
        return RNG.randint(-100000, 100000)
    elif itemtype is float:
        return round((RNG.random() - 0.5) * 200000, 2)
    elif itemtype is tuple:
        return tuple(
            round((RNG.random() - 0.5) * 200000, 2)
            for _ in range(RNG.randint(1, 10))
        )
    elif isinstance(itemtype, Callable):
        return itemtype()
    raise TypeError(f"no generator for {itemtype}")


def pick_key_at_level(mapping, keys, hashables, branch_weight):
    level_mapping = get_in(keys, mapping)
    nestkeys = [
        k for k, v in level_mapping.items() if isinstance(v, NestingDict)
    ]
    if (len(nestkeys) == 0) or (RNG.random() > branch_weight):
        nestkey = randval(RNG.choice(hashables))
    else:
        nestkey = RNG.choice(nestkeys)
    return nestkey


def random_nested_dict(
    quantity,
    maxdepth=4,
    types=(str, int, float, tuple),
    keytypes=None,
    branch_weight=0.4
):
    nest = NestingDict()
    levels = tuple(range(maxdepth))
    if keytypes is None:
        keytypes = [t for t in types if isinstance(t, Hashable)]
    for _ in range(quantity):
        level = RNG.choice(levels)
        keys = []
        for level_ix in range(level):
            keys.append(
                pick_key_at_level(nest, keys, keytypes, branch_weight)
            )
        get_in(keys, nest)[
            randval(RNG.choice(keytypes))
        ] = randval(RNG.choice(types))
    return nest.todict()


def piped(func):
    here, there = Pipe()

    def sendback(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as ex:
            result = ex
        return there.send(result)

    return here, sendback


def wrap_piped(func, block=True):
    def get_results_from_child_process(*args, **kwargs):
        here, sendback = piped(func)
        proc = Process(target=sendback, args=args, kwargs=kwargs)
        proc.start()
        if block is True:
            proc.join()
            result = here.recv()
            proc.close()
            return result
        return proc

    return get_results_from_child_process
