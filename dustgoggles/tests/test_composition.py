import re
import string
from pathlib import Path

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