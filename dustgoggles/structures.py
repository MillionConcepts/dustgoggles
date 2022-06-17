"""structured data and manipulators thereof"""
from collections import defaultdict
from copy import copy
from functools import reduce, partial
from operator import methodcaller, add, getitem, eq
from typing import (
    Mapping, Collection, Any, Union, Sequence, MutableMapping, Callable, Type
)

from cytoolz import merge

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


def unnest(mapping_mapping):
    unnested = []
    for category, mapping in mapping_mapping.items():
        unnested.append(
            {
                str(category) + "_" + str(key): value
                for key, value in mapping.items()
            }
        )
    return merge(unnested)


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
    mtypes: tuple[Type[MutableMapping]] = (dict,)
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
