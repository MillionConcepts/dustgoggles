import json
from itertools import accumulate
from operator import add
from pathlib import Path
from statistics import mean

from dustgoggles.structures import to_records, enumerate_as_mapping, \
    dig_for_value, dig_for_key, dig_for_values, get_from

# mars 2020 outreach waypoints file -- snapshot of
# https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json
test_file = Path(
    Path(__file__).parent.parent, 'test_utils', 'data', 'm20_waypoints.json'
)


def test_to_records():
    as_nested = {
        'April': {
            1: {'sales': 100, 'unit': 'million_usd'},
            2: {'sales': 110, 'unit': 'million_usd'},
            3: {'sales': 80, 'unit': 'million_usd'},
            4: {'sales': 90, 'unit': 'million_usd'}
        },
        'May': {
            1: {'sales': 130, 'unit': 'million_usd'},
            2: {'sales': 150, 'unit': 'million_usd'},
            3: {'sales': 60, 'unit': 'million_usd'},
            4: {'sales': 70, 'unit': 'million_usd'}
        }
    }
    records = to_records(as_nested, level_names=('month', 'week'))
    fourth_week_sales = map(
        lambda rec: rec['sales'],
        filter(lambda rec: rec['week'] == 4, records)
    )
    running_fourth_week_sales_total = accumulate(fourth_week_sales, add)
    assert next(running_fourth_week_sales_total) == 90
    assert next(running_fourth_week_sales_total) == 160
    try:
        next(running_fourth_week_sales_total)
        assert KeyError("that shouldn't have worked.")
    except StopIteration:
        pass


def test_dig_for():
    features = enumerate_as_mapping(json.load(test_file.open())['features'])
    assert dig_for_value(features, 'coordinates') == [
        77.45088572000003, 18.444627149999974, -2569.91
    ]
    assert dig_for_key(features, lambda _, v: v == 100, match="value") == 'sol'
    assert mean(dig_for_values(features, "lat")) - 18.4405 < 0.001


def test_get_from():
    class OstentatiouslyNestedObject:
        def __init__(self, interior=None, interior_name="interior"):
            if interior is not None:
                setattr(self, interior_name, interior)

    ostension = OstentatiouslyNestedObject(
        OstentatiouslyNestedObject(
            OstentatiouslyNestedObject(
                {
                    "1": 2,
                    "down": OstentatiouslyNestedObject(
                        {"cat": "tabby"}, "ababselectstart"
                    )
                }, "left"
            ), "right"
        ), "up"
    )

    assert get_from(
        ostension, ("up", "right", "left", "down", "ababselectstart", "cat")
    ) == "tabby"
