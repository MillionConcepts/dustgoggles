"""structured data and manipulators thereof"""
from collections import defaultdict, OrderedDict
from copy import copy
from functools import reduce, partial
from itertools import chain
from multiprocessing import Pool
from operator import methodcaller, add, getitem, eq
from typing import (
    Mapping,
    Collection,
    Any,
    Union,
    Sequence,
    MutableMapping,
    Callable,
    Type,
    Hashable,
)

from cytoolz import merge
from more_itertools import chunked, divide

from dustgoggles.func import naturals, is_it, splat


def to_records(nested: Mapping, accumulated_levels=None, level_names=None):
    level_names = naturals() if level_names is None else iter(level_names)
    records = []
    accumulated_levels = (
        {} if accumulated_levels is None else accumulated_levels
    )
    level_name = next(level_names)
    for category, mapping in nested.items():
        if all([isinstance(value, Mapping) for value in mapping.values()]):
            branch = accumulated_levels.copy()
            branch[level_name] = category
            records += to_records(mapping, branch, copy(level_names))
        else:
            category_dict = accumulated_levels | {level_name: category}
            flat = mapping | category_dict
            records.append(flat)

    return records


def _unnest(nested, mtypes, escape, levels):
    if not isinstance(nested, mtypes):
        return nested
    prefix = f"{escape}_{escape}".join(map(str, levels))
    if len(prefix) > 0:
        prefix = f"{escape}{prefix}{escape}_"
    long_records = []
    for level, maybe_mapping in nested.items():
        if not isinstance(maybe_mapping, mtypes):
            long_records.append({f"{prefix}{level}": maybe_mapping})
            continue
        long_records += _unnest(maybe_mapping, mtypes, escape,
                                levels + [level])
    return long_records


def unnest(nested, mtypes=(dict,), escape=""):
    flat_records = _unnest(nested, mtypes, escape, [])
    return merge(flat_records)


class NestingDict(defaultdict):
    """
    shorthand for automatically-nesting dictionary -- i.e.,
    insert a series of keys at any depth into a NestingDict
    and it automatically creates all needed levels above.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = NestingDict

    __repr__ = dict.__repr__

    def todict(self):
        """warning: this function may permanently modify lower levels"""
        # note: could be made depth-first by using methodcaller('todict')
        # rather than dict
        return dict(
            dig_and_edit(
                self, lambda _, v: is_it(NestingDict)(v), lambda _, v: dict(v)
            )
        )


def listify(thing):
    """Always a list, for things that want lists"""
    if isinstance(thing, Collection):
        if not isinstance(thing, str):
            return list(thing)
    return [thing]


def reindex_mapping(mapping: Mapping) -> Mapping[int, Any]:
    assert isinstance(mapping, Mapping), "reindex_mapping only takes Mappings"
    return {
        ix: value for ix, value in zip(range(len(mapping)), mapping.values())
    }


def enumerate_as_mapping(sequence):
    if sequence is None:
        return {}
    if isinstance(sequence, Mapping):
        return sequence
    return {ix: element for ix, element in enumerate(sequence)}


def insert_after(
    new_key: Any, new_value: Any, prior_key: Any, mapping: Mapping
) -> Mapping:
    new_dict = {}
    for key, value in mapping.items():
        new_dict[key] = value
        if key == prior_key:
            new_dict[new_key] = new_value
    return new_dict


def get_from_all(key, mappings, default=None):
    """
    get all values of "key" from each dict or whatever in "mappings"
    """
    if isinstance(mappings, Mapping):
        view = mappings.values()
    else:
        view = mappings
    return list(map(methodcaller("get", key, default), view))


def getitemattr(collection: Union[Sequence, Mapping], key: Any) -> Any:
    """
    getter that attempts both getattr and getitem (intended
    for named tuples nested inside of dicts, etc)
    """
    try:
        return getitem(collection, key)
    except (KeyError, IndexError, TypeError):
        return getattr(collection, key)


def get_from(collection, keys, default=None):
    """
    hierarchical (toolz-style) extension of get_itemattr() --
    (hierarchical list of keys, collection ->
    item of collection, possibly from a nested collection)
    """
    level = collection
    for key in keys:
        try:
            level = getitemattr(level, key)
        except AttributeError:
            return default
    return level


def constant_to_dig_predicate(putative_predicate, match="key", default=eq):
    if isinstance(putative_predicate, Callable):
        return putative_predicate
    if match == "key":
        return keyonly(partial(default, putative_predicate))
    elif match == "value":
        return valonly(partial(default, putative_predicate))
    elif match == "item":
        return partial(default, putative_predicate)


def keyonly(func):
    def onlykey(key, _value):
        return func(key)

    return onlykey


def valonly(func):
    def onlyval(_key, value):
        return func(value)

    return onlyval


def _evaluate_diglevel(mapping, predicate, mtypes):
    # note: this is a little awkward and is done to circumvent the fact
    # that checking isinstance against the ABCs is relatively slow
    # but we would like to support many types of Mapping
    # that are not dreamt of in this library.
    nests = [v for v in mapping.values() if isinstance(v, mtypes)]
    level_items = [(k, v) for k, v in mapping.items() if predicate(k, v)]
    return level_items, nests


def dig_for_all(mapping, predicate, mtypes):
    level_items, nests = _evaluate_diglevel(mapping, predicate, mtypes)
    if nests:
        level_items += reduce(
            add, [dig_for_all(nest, predicate, mtypes) for nest in nests]
        )
    return level_items


def dig_all_wrapper(mapping, ref, match, base_pred, element_ix, mtypes):
    predicate = constant_to_dig_predicate(ref, match, base_pred)
    items = dig_for_all(mapping, predicate, mtypes)
    if len(items) == 0:
        return None
    if element_ix is None:
        return items
    return [item[element_ix] for item in items]


def dig_for_values(mapping, ref, match="key", base_pred=eq, mtypes=(dict,)):
    return dig_all_wrapper(mapping, ref, match, base_pred, 1, mtypes)


def dig_for_keys(mapping, ref, match="key", base_pred=eq, mtypes=(dict,)):
    return dig_all_wrapper(mapping, ref, match, base_pred, 0, mtypes)


def dig_for_items(mapping, ref, match="key", base_pred=eq, mtypes=(dict,)):
    return dig_all_wrapper(mapping, ref, match, base_pred, None, mtypes)


def dig_for(mapping, predicate, mtypes):
    """
    return the first item, using a breadth-first search, matching ref
    """
    level_items, nests = _evaluate_diglevel(mapping, predicate, mtypes)
    if level_items:
        return level_items
    if not nests:
        return None
    dug, iternests = None, iter(nests)
    while dug is None:
        try:
            dug = dig_for(next(iternests), predicate, mtypes)
        except StopIteration:
            return None
    return dug


def dig_wrapper(mapping, ref, match, base_pred, element_ix, mtypes):
    predicate = constant_to_dig_predicate(ref, match, base_pred)
    item = dig_for(mapping, predicate, mtypes)
    if item is None:
        return None
    if element_ix is None:
        return item[0]
    return item[0][element_ix]


def dig_for_value(mapping, ref, match="key", base_pred=eq, mtypes=(dict,)):
    return dig_wrapper(mapping, ref, match, base_pred, 1, mtypes)


def dig_for_key(mapping, ref, match="key", base_pred=eq, mtypes=(dict,)):
    return dig_wrapper(mapping, ref, match, base_pred, 0, mtypes)


def dig_for_item(mapping, ref, match="key", base_pred=eq, mtypes=(dict,)):
    return dig_wrapper(mapping, ref, match, base_pred, None, mtypes)


def dig_and_edit(
    mapping: MutableMapping,
    filter_func: Callable[[Any, Any], Any],
    setter_func: Callable[[Any, Any], Any],
    mtypes: tuple[Type[MutableMapping]] = (dict,),
) -> MutableMapping:
    matches = tuple(filter(splat(filter_func), mapping.items()))
    for key, value in matches:
        mapping[key] = setter_func(key, value)
    for nest in [v for v in mapping.values() if isinstance(v, mtypes)]:
        dig_and_edit(nest, filter_func, setter_func, mtypes)
    return mapping


# TODO: turn pivot.split_on into a dispatch function in structures,
#  replace downstream
def separate_by(collection, ref):
    hits = []
    misses = []
    for item in collection:
        if ref(item) is True:
            hits.append(item)
        else:
            misses.append(item)
    return hits, misses


def rmerge(map1, map2, mtypes=(dict, OrderedDict)):
    """
    recursively merge map1 and map2. result is similar to map1 | map2, but
    also merges any mappings at lower levels. for example:

    >>> rmerge({'a': 1, 'b': {'a': 1}}, {'b': {'b': 1}, 'c': 2})
    {'a': 1, 'b': {'a': 1, 'b': 1}, 'c': 2}
    """
    output = {}
    only_in_one = set(map1.keys()).symmetric_difference(map2.keys())
    for kv in chain(map1.items(), map2.items()):
        if kv[0] in only_in_one:
            output[kv[0]] = kv[1]
        elif all(map(lambda m: isinstance(m, mtypes), (map1[kv[0]], map2[kv[0]]))):
            output[kv[0]] = rmerge(map1[kv[0]], map2[kv[0]])
        else:
            output[kv[0]] = map2[kv[0]]
    return output


class HashDict:
    def __init__(self, equivalence: Callable[[Any], Hashable] = hash):
        self.dict_ = {}
        self.reverse = {}
        self.hasher = equivalence

    def __setitem__(self, key, item):
        itemhash = self.hasher(item)
        self.dict_[key] = itemhash
        if itemhash not in self.reverse.keys():
            self.reverse[itemhash] = item

    def __getitem__(self, key):
        return self.reverse[self.dict_[key]]

    def __str__(self):
        return (
            f"HashDict with {len(self.dict_)} keys storing "
            f"{len(self.reverse)} unique values"
        )

    def __repr__(self):
        return self.__str__()


class MaybeResult:
    def __init__(self, func, args, kwargs):
        try:
            self.value = func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            self.value = ex

    def get(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value

    @staticmethod
    def ready():
        return True


def _chunkwrap(func, argrecs):
    return {
        rec['key']: func(*rec['args'], **rec['kwargs'])
        for rec in argrecs
    }


class MaybePool:
    def __init__(self, threads=None):
        self.threads, self.results, self.chunked = threads, {}, False
        if threads is None:
            self.pool = None
        else:
            self.pool = Pool(threads)

    def map(self, func, argrecs, as_chunks=False):
        for i, rec in enumerate(argrecs):
            rec['key'] = rec.get('key', i)
            rec['args'] = rec.get('args', ())
            rec['kwargs'] = rec.get('kwargs', {})
        if (as_chunks is False) or (self.pool is None):
            self.results = {
                rec['key']: self.apply(func, rec['args'], rec['kwargs'])
                for rec in argrecs
            }
        else:
            self.chunked = True
            chunks = divide(self.threads, argrecs)
            self.results = {
                f"chunk_{i}": self.apply(_chunkwrap, (func, chunk))
                for i, chunk in enumerate(chunks)
            }

    def apply(self, func, args=(), kwargs=None):
        """note: does apply_async by default"""
        kwargs = {} if kwargs is None else kwargs
        if self.pool is None:
            return MaybeResult(func, args, kwargs)
        return self.pool.apply_async(func, args, kwargs)

    def close(self):
        if self.pool is None:
            return
        return self.pool.close()

    def join(self):
        if self.pool is None:
            return
        return self.pool.join()

    def get(self, raise_exc=False):
        output = {}
        for k, v in self.results.items():
            try:
                if self.chunked is True:
                    output |= v.get()
                else:
                    output[k] = v.get()
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                if raise_exc:
                    raise ex
                output[k] = ex
        return output

    def results_ready(self):
        return {k: v.ready() for k, v in self.results.items()}

    def ready(self):
        return all(v.ready() for v in self.results.values())

    def terminate(self):
        if self.pool is not None:
            self.pool.terminate()
        self.results = {}


def map_into_pool(
    func,
    argrecs,
    threads=None,
    filter_exc=False,
    as_chunks=True
):
    pool = MaybePool(threads)
    try:
        pool.map(func, argrecs, as_chunks=as_chunks)
        pool.close()
        pool.join()
        results = pool.get()
        if filter_exc is True:
            results = {
                k: v for k, v in results.items()
                if not isinstance(v, Exception)
            }
    finally:
        pool.terminate()
    return results
