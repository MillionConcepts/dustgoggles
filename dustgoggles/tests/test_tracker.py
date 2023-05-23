from dustgoggles.composition import Composition
from dustgoggles.tracker import Tracker

SUMMARY = """
Characterizing Positional Offsets in Map-Projected Mini-RF Monostatic Data 
[#2856] This abstract is an update and summary of continuing efforts to correct 
positional errors and reprocess map-projected Mini-RF monostatic data products.
"""
EXPECTED_STEMS = {
    '', 'abstract', 'monostatic', 'correct', 'offsets', 'this', 'to',
    'update', 'map-projected', 'positional', 'products.', 'continuing',
    'mini-rf', 'summary', '[#2856]', 'errors', 'is', 'and', 'characterizing',
    'data', 'reprocess', 'of', 'an', 'efforts', 'in'
}


def test_tracker():
    steps = {
        "strip": str.strip,
        "oneline": lambda a: a.replace("\n", " "),
        "lower": str.lower,
        "split": lambda a: a.split(" "),
        "unique": set
    }
    stemmer = Composition(steps, tracker=Tracker(), name='stemmer')
    stemmed = stemmer.execute(SUMMARY)
    assert stemmed == EXPECTED_STEMS
    assert stemmer.tracker.history[0]['target'] == 'strip'
    assert stemmer.tracker.history[1]['step'] == 'oneline'
