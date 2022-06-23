from operator import add, mul, sub
from pathlib import Path
import re
import string

from cytoolz import frequencies, curry, get

from dustgoggles.composition import Composition


def slurp(path):
    with open(path) as stream:
        return stream.read()


def strip_unprintable(text):
    return "".join([c for c in text if c in string.printable])


test_file = Path(
    Path(__file__).parent.parent, 'test_utils', 'data', '1342-0.txt'
)


def test_composition_equivalence():
    or_punctuation = "|".join([
        re.escape(char) for char in string.punctuation
    ])
    punctuation = rf"({or_punctuation})"
    inserted = Composition(
        steps={
            'slurp': slurp,
            'lower': str.lower,
            'strip': strip_unprintable,
            'depunctuate': re.sub,
            'split': re.split,
            'count': frequencies
        },
        inserts={
            'depunctuate': {0: punctuation, 1: ""},
            'split': {0: r"(,|\s)"}
        }
    )

    # noinspection PyArgumentList
    partially = Composition(
        steps={
            'slurp': slurp,
            'lower': str.lower,
            'strip': strip_unprintable,
            'depunctuate': curry(re.sub)(punctuation, ""),
            'split': curry(re.split)(r"(,|\s)"),
            'count': frequencies
        }
    )

    result_0 = inserted.execute(test_file)
    result_1 = partially.execute(test_file)
    assert result_0 == result_1
    assert get(
        ['truth', 'universally', 'acknowledged'], result_0
    ) == (27, 3, 20)


def test_composition_ordering():
    composition = Composition((add, mul, sub))
    composition.add_insert(0, 0, 1)
    composition.add_insert(1, 0, 2)
    composition.add_send(0, None, 2, 0)
    # 3 - (2 + 1) * 2
    assert composition.execute(2) == -3
    # clear bus
    composition.sends[0] = []
    composition.inserts[2] = {}
    composition.add_send(0, None, 2, 1)
    # (2 + 1) * 2 - 3
    assert composition.execute(2) == 3
