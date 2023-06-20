from itertools import count
from operator import attrgetter
from typing import (
    Optional,
    Union,
    Mapping,
    Any,
    Callable,
    Sequence,
    Hashable,
    MutableSequence,
    Literal,
)

from cytoolz import identity, keyfilter, first

from dustgoggles.dynamic import exc_report
from dustgoggles.func import naturals
from dustgoggles.structures import enumerate_as_mapping, reindex_mapping
from dustgoggles.tracker import TrivialTracker, Tracker


class Composition:
    """
    class defining a composition of steps, optionally with
    additional input, output, and i/o points ("inserts" and "sends") --
    conceptually like a signal processing chain but _not_ designed for
    real-time signal processing.
    """

    def __init__(
        self,
        steps: Optional[
            Union[Mapping[Any, Callable], Sequence[Callable]]
        ] = None,
        sends: Optional[Union[Mapping[Any, Sequence], Sequence]] = None,
        inserts: Optional[
            Union[Mapping[Any, Mapping], Sequence[Mapping]]
        ] = None,
        patch: Optional[Union[Mapping[Any, Sequence], Sequence]] = None,
        name: Optional[str] = None,
        tracker: Optional[TrivialTracker] = None,
        optional: bool = False,
    ):
        self.captures = None
        self.name = "untitled Composition" if name is None else name
        if tracker is not None:
            tracker.name = self.name
        self.tracker = tracker
        if steps is None:
            steps = [identity]
        self.steps = enumerate_as_mapping(steps)
        self.tracker = tracker
        # sends and inserts could be defaultdicts, but in this case the
        # representation is ugly and the convenience is small.
        self.sends = enumerate_as_mapping(sends)
        self.inserts = enumerate_as_mapping(inserts)
        self.patch = enumerate_as_mapping(patch)
        self.optional = optional
        if self.tracker is not None:
            self._add_track_sends()

    def track(self):
        if self.tracker is not None:
            return
        self.tracker = Tracker()
        self._add_track_sends()

    def _check_for_step(self, step_name: Hashable) -> Any:
        if step_name not in self.steps.keys():
            raise KeyError(f"{step_name} is not an element of the pipeline.")
        return self.steps[step_name]

    def add_step(self, step: Callable, name: Optional[Hashable] = None):
        """
        add step with name "name" -- by default, length of the pipeline + 1.
        if no step with that name currently exists, add it to the end of the
        pipeline; otherwise, replace the current step at "name".
        """
        if name is None:
            name = len(self.steps) + 1
        self.steps[name] = step
        self._add_track_send(name)

    def _add_track_sends(self):
        for k in self.steps.keys():
            self._add_track_send(k)

    def _add_track_send(self, step_name: Hashable):
        if self.tracker is None:
            return
        self.add_send(
            step_name,
            pipe=_track_factory(
                self.tracker, step_name, self.steps[step_name]
            ),
        )

    def reindex(self):
        self.steps = reindex_mapping(self.steps)

    @property
    def index(self):
        return tuple(self.steps.keys())

    @property
    def insert_index(self):
        return tuple(self.inserts.keys())

    @property
    def send_index(self):
        return tuple(self.sends.keys())

    @property
    def function_names(self):
        return list(map(attrgetter("__name__"), self.steps.values()))

    def _perform_sends(self, step_name: Hashable, state: Any):
        """
        send pipeline state to send targets, if one matching this step name
        exists. unlike inserts, there is currently no provision for sends with
        value None; there probably should not be.
        """
        buses = self.sends.get(step_name)
        if (buses is None) or (len(buses) == 0):
            return
        args, kwargs, unpack = self._route_signal(state, step_name)
        dopass = (len(args) + len(kwargs)) == 0
        for pipe, target, index in buses:
            if target is None:
                if pipe is None:
                    continue
                self._flow_signal(pipe, state, args, kwargs, unpack, dopass)
            elif pipe is None:
                self.place_into(state, target, index)
            else:
                self.place_into(
                    self._flow_signal(
                        pipe, state, args, kwargs, unpack, dopass
                    ),
                    target,
                    index
                )

    def add_captures(self):
        """add captures to all steps."""
        self.captures = {step: [None] for step in self.steps}
        for k, v in self.captures.items():
            self.add_capture(k, v)

    def add_capture(
        self, step_name: Hashable, target: Optional[MutableSequence]
    ):
        """
        creates a send to a simple capture object that stores the last output
        of that pipeline step.
        """
        target = [None] if target is None else target
        self.add_send(step_name, target=target, pointer=0)

    def add_send(
        self,
        step_name: Hashable,
        pipe: Optional[Callable] = None,
        target: Optional[Any] = None,
        pointer: Union[str, int, None] = None,
    ):
        """
        adds a send to the pipeline after step_name; sends to pointer
        at target after processing through pipe. a string or int is an
        insert into the step named 'target' of self. if no target is given,
        merely calls pipe and sends its output to nowhere. you are allowed
        to create sends from nonexistent step names, but they will do nothing
        unless that step name is later assigned.
        """
        if (target is None) and (pipe is None):
            raise ValueError(
                "Either pipe or target must be defined to create a send."
            )
        if step_name not in self.sends.keys():
            self.sends[step_name] = []
        self.sends[step_name].append((pipe, target, pointer))

    def add_insert(
        self, step_name: Hashable, key: Union[str, int], value: Any
    ):
        """
        adds an insert to the pipeline at step_name. an integer-valued
        "key" will turn into a positional argument to step; a string-valued
        "key" will turn into a keyword argument to the step at step name.
        "value" is simply the value of that arg or kwarg. you are allowed to
        add inserts to nonexistent steps, but they will do nothing unless that
        step name is later assigned.
        """
        if step_name not in self.inserts.keys():
            self.inserts[step_name] = {}
        self.inserts[step_name][key] = value

    # TODO: evaluate whether it might be important to base this tree on checks
    #  against the ABCs, which are of course unfortunately expensive -- but
    #  only trivally if this is being checked less than ~30 times per second.
    #  perhaps find some way to modify it at runtime.
    def place_into(self, thing, target, pointer):
        if isinstance(target, (str, int)):
            self.add_insert(target, pointer, thing)
        elif isinstance(target, list):
            if pointer is None:
                target.append(thing)
            else:
                target[pointer] = thing
        elif isinstance(target, dict):
            target[pointer] = thing
        elif "write" in dir(target):
            target.write(thing)

    @staticmethod
    def _flow_args(state, args):
        # signal flows into the first open position, permitting partial
        # evaluation of steps with 'fixed' positional-only arguments
        opening = first(
            filter(lambda integer: integer - 1 not in args.keys(), naturals())
        )
        args[opening - 1] = state
        return [args[k] for k in sorted(args.keys())]

    @staticmethod
    def _flow_into_partial(step, args, kwargs):
        # note that inserts / routes always override args and kwargs baked
        # into the step itself.
        if "args" in dir(step):
            args = enumerate_as_mapping(step.args) | args
        if "keywords" in dir(step):
            kwargs = step.keywords | kwargs
        return step.func, args, kwargs

    def _route_signal(
        self, result, prior, which: Literal["step", "send"] = "step"
    ):
        if prior is None:
            return {}, {}, None
        if self.patch.get(prior) is None:
            return {}, {}, None
        if (routespec := self.patch[prior].get(which)) is None:
            return {}, {}, None
        args, kwargs, unpack = {}, {}, routespec.get("unpack")
        if (argmap := routespec.get("argmap")) is not None:
            args = {m: result[n] for m, n, in argmap.items()}
        if (kwargmap := routespec.get("kwargmap")) is not None:
            kwargs = {m: result[n] for m, n in kwargmap.items()}
        return args, kwargs, unpack

    def _flow_signal(self, step, state, args, kwargs, unpack, dopass):
        if "func" in dir(step):
            step, args, kwargs = self._flow_into_partial(step, args, kwargs)
        if (len(args) != 0) and (dopass is True):
            dopass, args = False, self._flow_args(state, args)
        elif len(args) == 0:
            args = ()
        else:
            args = [args[k] for k in sorted(args.keys())]
        if dopass is False:
            return step(*args, **kwargs)
        if unpack is None:
            return step(state, *args, **kwargs)
        elif unpack == "*":
            return step(*state, *args, **kwargs)
        elif unpack == "**":
            return step(*args, **kwargs, **state)
        raise ValueError("bad unpacking specification")

    def call_step(self, step_name, state, prior):
        rargs, rkwargs, unpack = self._route_signal(state, prior)
        dopass = (len(rargs) + len(rkwargs)) == 0
        step, insert = self.steps[step_name], self.inserts.get(step_name)
        if insert is not None:
            iargs = keyfilter(lambda k: isinstance(k, int), insert)
            ikwargs = keyfilter(lambda k: isinstance(k, str), insert)
            if match := set(iargs).intersection(rargs):
                raise ValueError(f"arg overpatch on {step_name}: {match}")
            if match := set(ikwargs).intersection(rkwargs):
                raise ValueError(f"kwarg overpatch on {step_name}: {match}")
        else:
            iargs, ikwargs = {}, {}
        args, kwargs = rargs | iargs, rkwargs | ikwargs
        return self._flow_signal(step, state, args, kwargs, unpack, dopass)

    def _do_step(self, step_name, state, prior=None):
        """perform an individual step of the pipeline"""
        state = self.call_step(step_name, state, prior)
        self._perform_sends(step_name, state)
        return state

    def itercall(self, signal: Any = None, **special_kwargs):
        self._bind_special_runtime_kwargs(special_kwargs)
        state, prior = signal, None
        for step_name in self.steps.keys():
            state, prior = self._do_step(step_name, state, prior), step_name
            yield state

    def execute(self, signal: Any = None, **special_kwargs):
        """execute the pipeline, initializing it with signal."""
        step_ix = -1
        try:
            iterpipe = self.itercall(signal, **special_kwargs)
            state = None
            for step_ix, state in zip(count(), iterpipe):
                pass
            return state
        except Exception as ex:
            if self.tracker is not None:
                self.tracker.track(
                    self.steps[tuple(self.steps.keys())[step_ix + 1]],
                    **exc_report(ex, stepback=0),
                )
            if self.optional is True:
                return None
            raise

    def iter(self):
        raise NotImplementedError

    def _bind_special_runtime_kwargs(self, special_kwargs):
        """
        apply kwargs at time of execution across multiple steps
        according to domain-specific business logic.
        this exists for use by subclasses. it does nothing
        in the base Composition class.
        """
        pass

    def __str__(self):
        me_string = ""
        for attr in ("steps", "sends", "inserts", "route"):
            if attr not in dir(self):
                continue
            if len(getattr(self, attr)) == 0:
                continue
            me_string += f"{attr}:\n{getattr(self, attr).__repr__()}\n"
        if me_string == "":
            return "empty Composition"
        return me_string

    def __repr__(self):
        return self.__str__()

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)


def _track_factory(tracker, step, func):
    def track(_):
        return tracker.track(func, step=step)

    return track
