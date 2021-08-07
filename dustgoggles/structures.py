"""structured data and manipulators thereof"""
from collections import defaultdict
from copy import copy
from operator import methodcaller
from typing import Mapping, Collection, Any

from cytoolz import merge

from dustgoggles.func import naturals


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