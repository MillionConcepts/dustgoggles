"""functional utilities and generators"""
from functools import wraps, partial, reduce
from itertools import accumulate, repeat
from operator import add, contains, and_, getitem
from typing import Callable, Iterable, Any, Mapping


def pass_parameters(func, *args, **kwargs):
    return func(*args, **kwargs)


def catch_interaction(noninteractive, func, *args, **kwargs):
    """
    if noninteractive is truthy, always return empty string. intended
    primarily as a wrapper to preempt attempts to prompt user input.
    """
    if noninteractive:
        return ""
    return func(*args, **kwargs)


def naturals():
    return accumulate(repeat(1), add)


def zero(*_, **__):
    """take anything, return nothing"""
    return


def triggerize(niladic_function: Callable) -> Callable:
    """
    implicitly turns a function into a trigger step for
    a pipeline
    """

    @wraps(niladic_function)
    def trigger(deferred_state=None):
        niladic_function()
        return deferred_state

    return trigger


def are_in(items: Iterable, oper: Callable = and_) -> Callable:
    """
    iterable -> function
    returns function that checks if its single argument contains all
    (or by changing oper, perhaps any) items
    """

    def in_it(container: Iterable) -> bool:
        inclusion = partial(contains, container)
        return reduce(oper, map(inclusion, items))

    return in_it


def is_it(*types: type) -> Callable[[Any], bool]:
    """partially-evaluated predicate form of `isinstance`"""

    def it_is(whatever: Any):
        return isinstance(whatever, types)

    return it_is


def intersection(*iterables):
    return reduce(and_, [set(iterable) for iterable in iterables])


def disjoint(*sets):
    shared = intersection(*sets)
    return [
        [file for file in this_set if file not in shared] for this_set in sets
    ]


def constant(value):
    def return_constant(*_, **__):
        return value

    return return_constant
