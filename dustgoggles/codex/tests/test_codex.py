import random
import time

from dustgoggles.codex.implements import Notepad, Sticky
from dustgoggles.test_utils import wrap_piped


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


def look_up_kwargs(func):
    def kwarg_lookup(notepad_title, key):
        return func(**Notepad(notepad_title).get(key))

    return kwarg_lookup


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


def test_notepad_1():
    address = random.randint(100000, 2000000)
    remote_square = wrap_piped(look_up_kwargs(square))
    notepad = Notepad(f"test_notepad_{address}", create=True)
    notepad["cat"] = {"x": 2}
    notepad["dog"] = {"x": 3}
    assert remote_square(f"test_notepad_{address}", "cat") == 4
    assert remote_square(f"test_notepad_{address}", "dog") == 9
