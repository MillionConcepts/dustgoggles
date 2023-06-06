import datetime as dt
import json
from inspect import currentframe, getframeinfo
from itertools import count
from pathlib import Path


class TrivialTracker:
    def dump(self):
        pass

    def track(self, *_, **__):
        pass

    def clear(self):
        pass

    def set_metadata(self, **metadata):
        pass

    log = None
    history = None


class Tracker(TrivialTracker):
    """watches where it's been"""

    def __init__(self, name=None, identifier="", outdir=None):
        self.history, self.metadata, self.counter = [], {}, count(1)
        self.name = name
        self.log = {
            "name": self.name,
            "init_timestamp": dt.datetime.now().isoformat(),
            "history": self.history,
        }
        if outdir is None:
            outdir = Path(__file__).parent.parent / ".tracker_logs"
        if identifier != "":
            identifier = f"{identifier}_"
        outdir.mkdir(exist_ok=True)
        self.outpath = Path(outdir, f"{identifier}{self.name}.json")

    def clear(self):
        self.outpath.unlink(missing_ok=True)
        self.history = []
        self.counter = count(1)

    def track(self, func, **metadata):
        if "__name__" in dir(func):
            target = func.__name__
        else:
            target = str(func)
        caller = currentframe().f_back
        info = getframeinfo(caller)
        rec = (
            {
                "target": target,
                "caller": info.function,
                "lineno": info.lineno,
                "trackcount": next(self.counter),
            }
            | self.metadata
            | metadata
        )
        self.history.append(rec)

    def set_metadata(self, **metadata):
        self.metadata |= metadata

    def _get_paused(self):
        return self._paused

    def _set_paused(self, state: bool):
        self._paused = state

    def dump(self):
        if self.paused is True:
            return
        # TODO: do this more cleverly with incremental writes etc.
        self.log["write_timestamp"] = dt.datetime.now().isoformat()
        key_order = sorted(self.log.keys(), key=lambda n: "history" in n)
        self.log = {k: self.log[k] for k in key_order}
        with self.outpath.open("a") as stream:
            json.dump(self.log, stream, indent=2)

    _paused = False
    paused = property(_get_paused, _set_paused)
