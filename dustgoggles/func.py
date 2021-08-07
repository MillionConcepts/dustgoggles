"""functional utilities and generators"""
from functools import wraps
from itertools import accumulate, repeat
from operator import add
from typing import Callable


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
