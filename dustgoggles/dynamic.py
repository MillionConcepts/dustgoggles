"""utilities for dynamic function generation and code introspection"""
from inspect import getsource, signature, Signature, getmodule
from types import FunctionType, CodeType, MethodType
from typing import Optional, Callable, MutableSequence

from cytoolz import nth

from dustgoggles.func import optionalize


# TODO: some kind of "FunctionLike" type

class AlreadyLoadedError(ValueError):
    pass


class UnreadyError(ValueError):
    pass


# TODO, maybe: optimization stuff re: kwarg mapping, etc. -- but if you wanted
#  to engage in really high call volume, you could also just explicitly call
#  the wrapped function...
def digsource(obj: Callable):
    """
    wrapper for inspect.getsource that attempts to work on objects like
    Dynamic, functools.partial, etc.
    """
    if isinstance(obj, (FunctionType, MethodType)):
        return getsource(obj)
    if "func" in dir(obj):
        # noinspection PyUnresolvedReferences
        return getsource(obj.func)
    raise TypeError(f"cannot get source for type {type(obj)}")


def get_codechild(code: CodeType, ix: int = 0) -> CodeType:
    return nth(ix, filter(lambda c: isinstance(c, CodeType), code.co_consts))


def dontcare(func: Callable, target: MutableSequence) -> Callable:
    def record_exception(exc):
        target.append(exc_report(exc) | {'func': func, 'category': 'call'})

    return optionalize(func, exc_callback=record_exception)


def compile_source(source: str):
    return get_codechild(compile(source, "", "exec"))


def define(code: CodeType, globals_: Optional[dict] = None) -> FunctionType:
    globals_ = globals_ if globals_ is not None else globals()
    return FunctionType(code, globals_)


def exc_report(exc, stringify_exception=True):
    if exc is None:
        return {}
    tb = exc.__traceback__
    exc = exc if stringify_exception is False else f"{type(exc)}: {exc}"
    report = {'exception': exc, 'lineno': [], 'name': [], 'filename': []}
    while tb is not None:
        fr = tb.tb_frame
        if (module := getmodule(fr)) is not None:
            try:
                codename = fr.f_code.co_qualname
            except AttributeError:
                codename = fr.f_code.co_name
            report['name'].append(f'{module.__name__}.{codename}')
        else:
            report['name'].append(fr.f_code.co_name)
        report['lineno'].append(tb.tb_lineno)
        report['filename'].append(fr.f_code.co_filename)
        del fr  # taking excessive care here
        tb = tb.tb_next
    return report


class Dynamic:
    """
    simple class to help manage function definition / execution from
    dynamically-generated source.
    """
    def __init__(
        self,
        source: Optional[str] = None,
        globals_: Optional[dict] = None,
        optional: bool = False,
        lazy: bool = False,
        load_on_call: bool = True
    ):
        self.globals_ = globals_
        self.optional = optional
        self.errors = []
        self.load_on_call = load_on_call
        self.lazy = lazy
        self.call_fail = False
        self.compile_fail = False
        self.source = source
        if lazy is False:
            try:
                self.load()
            # should only encounter this when called from a class constructor
            except AlreadyLoadedError:
                pass

    # TODO: deal with modules not imported at compile-time
    #  -- maybe just permit second compilation --
    #  -- or do we need to pass globals? ugh.

    def load(self, reload=False):
        if (reload is False) and (self.func is not None):
            raise AlreadyLoadedError("self.func already loaded")
        self.compile_source(reload)
        self.define(reload)

    def compile_source(self, recompile=True):
        if (recompile is False) and (self.code is not None):
            raise AlreadyLoadedError("self.code already compiled")
        try:
            self.code = compile_source(self.source)
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            self.errors.append(exc_report(ex) | {'category': 'compile'})
            self.compile_fail = True

    def define(self, redefine=False):
        if (self.func is not None) and (redefine is not True):
            raise AlreadyLoadedError("self.func already defined")
        self.func = define(self.code, self.globals_)
        self.__signature__ = signature(self.func)
        self.__name__ = self.func.__name__

    def unload(self):
        del self.code, self.func, self.errors
        self.call_fail, self.compile_fail = False, False
        self.code, self.func, self.errors = None, None, []
        self.__name__ = self.__class__.__name__
        self.__signature__ = None

    def _maybe_load_on_call(self, reload=False):
        if self.func is not None:
            return
        if self.load_on_call is True:
            return self.load(reload)
        raise UnreadyError("No loaded function.")

    def __call__(self, *args, _optional=None, **kwargs):
        self._maybe_load_on_call()
        if _optional is None:
            _optional = self.optional
        if _optional is False:
            # noinspection PyUnresolvedReferences
            return self.func(*args, **kwargs)
        try:
            return dontcare(self.func, self.errors)(*args, **kwargs)
        finally:
            if len(self.errors) > 0:
                if self.errors[-1].get('category') == 'call':
                    self.call_fail = True

    def __str__(self):
        if self.func is None:
            return self.__class__.__name__
        return f"{self.__class__.__name__} {signature(self.func)}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_function(cls, func: Callable, *init_args, **init_kwargs):
        # TODO: some of this boilerplate could be reduced with a bunch of
        #  fancy getattr overrides
        dynamic = super().__new__(cls)
        try:
            dynamic.source = getsource(func)
        except TypeError:  # it's some kind of weirdo callable
            # noinspection PyUnresolvedReferences
            func = func.__call__
            dynamic.source = getsource(func)
        dynamic.code = func.__code__
        dynamic.func = func
        dynamic.__signature__ = signature(dynamic.func)
        dynamic.__name__ = dynamic.func.__name__
        dynamic.globals_ = func.__globals__
        dynamic.__init__(*init_args, **init_kwargs)
        return dynamic

    __signature__ = Signature()
    source, code, func, __name__ = None, None, None, '<unloaded Dynamic>'
