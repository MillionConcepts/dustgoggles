import random
import time
from multiprocessing import Pipe, Process

import pytest

from dustgoggles.codex.implements import Sticky


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


def square(x):
    return x**2


def stick_kwargs(func):
    def kwarg_stick(sticky_title):
        return func(**Sticky(sticky_title).read())

    return kwarg_stick


def stick_args(func):
    def arg_stick(sticky_title):
        return func(*Sticky(sticky_title).read())

    return arg_stick


def stuck_where(obj):
    return Sticky.note(obj).address


def delayed_stick(obj, address, delay):
    time.sleep(delay)
    Sticky.note(obj, address)


def delayed_read(address, delay):
    time.sleep(delay)
    return Sticky(address).read()


def test_sticky_1():
    remote_square = wrap_piped(stick_kwargs(square))
    assert isinstance(remote_square("empty address"), TypeError)


def test_sticky_2():
    remote_square = wrap_piped(stick_kwargs(square))
    sticky = Sticky.note({"x": 12})
    assert remote_square(sticky.address) == 144


def test_sticky_3():
    address = wrap_piped(stuck_where)({"x": 12})
    remote_square = wrap_piped(stick_kwargs(square))
    assert remote_square(address) == 144


def test_sticky_4():
    address = random.randint(100000, 2000000)
    wrap_piped(delayed_stick, block=False)(12, address, 0.1)
    assert wrap_piped(delayed_read)(address, 0.15) == 12
