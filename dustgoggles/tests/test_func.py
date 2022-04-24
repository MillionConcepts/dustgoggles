import random
from functools import partial
from operator import add, sub, or_

from dustgoggles.func import (
    catch_interaction, pass_parameters, naturals, zero, triggerize, are_in, is_it, intersection, disjoint, constant,
    splat
)


def test_pass_parameters():
    x, y = random.randint(1, 100), random.randint(1, 100)
    assert pass_parameters(add, x, y) == x + y
    assert pass_parameters(sub, x, y) == x - y


def test_catch_interaction():
    catch = partial(catch_interaction, True)
    dont = partial(catch_interaction, False)

    def shout():
        return random.randint(1, 1000)

    assert catch(shout) == ""
    assert isinstance(dont(shout), int)


def test_naturals():
    doublestruck_n = naturals()
    for i in range(1, 100):
        assert next(doublestruck_n) == i


def test_zero():
    for obj in locals().values():
        assert zero(obj) is None


def test_constant():
    one = constant(1)
    for obj in locals().values():
        assert one(obj) == 1


def test_triggerize():
    global_mutable = []

    def terrible_side_effect():
        global_mutable.append(random.randint(1, 100))

    terrible_trigger = triggerize(terrible_side_effect)
    i = 1
    for _ in range(10):
        i += terrible_trigger(i)
    assert i == 1024
    assert len(global_mutable) == 10


def test_are_in():
    query = {1, "cat", ("a", "B")}
    reference_1 = (4, "dog", ("a", "B"))
    reference_2 = (1, 4, "dog", "cat", ("a", "B"))
    all_are_in = are_in(query)
    any_are_in = are_in(query, oper=or_)
    assert not all_are_in(reference_1)
    assert all_are_in(reference_2)
    assert any_are_in(reference_1)
    assert any_are_in(reference_2)


def test_is_it():
    it_is = is_it(int, str, dict)
    assert it_is({"a": 1})
    assert it_is(2)
    assert not it_is(float)
    assert not it_is(it_is)


def test_intersection():
    sequences = (
        (0, 1, 2), [1, 2, 3], {2: "2", 3: "3", 4: "4"},
    )
    assert intersection(*sequences) == {2}


def test_disjoint():
    sequences = (
        (0, 1, 2), [1, 2, 3], {2: "2", 3: "3", 4: "4"},
    )
    assert disjoint(*sequences) == [[0, 1], [1, 3], [3, 4]]


# noinspection PyArgumentList
def test_splat():
    args = (2, 3, 4)

    def splattable(x, y, z):
        return x ** 3 + 2 * y ** 2 + 3 * z

    try:
        splattable(args)
        raise TypeError
    except TypeError:
        pass
    assert splat(splattable)(args) == 38

