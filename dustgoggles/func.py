"""functional utilities and generators"""
from functools import wraps, partial, reduce
from itertools import accumulate, repeat
from operator import add, contains, and_
from typing import Callable, Iterable, Any, Sequence, Collection


def pass_parameters(func, *args, **kwargs):
    return func(*args, **kwargs)


def catch_interaction(
    noninteractive: Any, func: Callable, *args, default: Any = "", **kwargs
):
    """
    if noninteractive is truthy, always return default. intended
    primarily as a wrapper to preempt attempts to prompt user input.
    """
    if noninteractive:
        return default
    return func(*args, **kwargs)


def naturals() -> accumulate:
    return accumulate(repeat(1), add)


def zero(*_, **__) -> None:
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


def are_in(items: Collection, oper: Callable = and_) -> Callable:
    """
    iterable -> function
    returns function that checks if its single argument contains all
    (or by changing oper, perhaps any) items
    """

    def in_it(container: Collection) -> bool:
        inclusion = partial(contains, container)
        return reduce(oper, map(inclusion, items))

    return in_it


def is_it(*types: type) -> Callable[[Any], bool]:
    """partially-evaluated predicate form of `isinstance`"""

    def it_is(whatever: Any):
        return isinstance(whatever, types)

    return it_is


def intersection(*iterables: Iterable) -> set:
    return reduce(and_, [set(iterable) for iterable in iterables])


def disjoint(*sets: Iterable) -> list[list]:
    shared = intersection(*sets)
    return [
        [file for file in this_set if file not in shared] for this_set in sets
    ]


def constant(value: Any) -> Callable:
    def return_constant(*_, **__):
        return value

    return return_constant


def splat(func: Callable) -> Callable[[Sequence], Any]:
    @wraps(func)
    def splatified(args: Sequence) -> Any:
        return func(*args)

    return splatified
