import json
from pathlib import Path

import pytest

from dustgoggles.pivot import extract_constants, itemize_numpy, pdstr, split_on

pd = pytest.importorskip('pandas')
np = pytest.importorskip('numpy')

test_file = Path(
    Path(__file__).parent.parent, 'test_utils', 'data', 'sorghum_forecast.csv'
)


def test_extract_constants():
    df = pd.read_csv(test_file)
    constants, variables = extract_constants(df)
    assert isinstance(constants, dict)
    assert constants == {
        'Commodity2': 'Sorghum',
        'Attribute': 'Beginning stocks',
        'Unit': 'Million bushels',
        'YearProjected2': 2022
    }
    assert variables['Value'].sum() == 358


def test_itemize_numpy():
    void_array = np.array([b"bbbbb"], dtype="V")
    assert void_array.dtype.name == "void40"
    assert isinstance(itemize_numpy(void_array[0]), bytes)
    int_array = np.array([0])
    assert int_array[0].dtype == np.dtype('int64')
    try:
        json.dumps(int_array[0])
        raise TypeError("that should not have worked.")
    except TypeError:
        pass
    assert json.dumps(itemize_numpy(int_array[0])) == "0"


def test_pdstr():
    df = pd.read_csv(test_file)
    extractor = pdstr("extract", r"s(\w)(\w)")
    answers = iter([np.nan, ["t", "o"], ["h", "e"], np.nan])

    for name, col in df.iteritems():
        if not col.dtype == np.dtype("O"):
            continue
        extracted = extractor(col)
        answer = next(answers)
        if isinstance(answer, list):
            assert np.all(answer == extracted.values[0])
        else:
            assert not np.isfinite(extracted.values[0][0])


def test_split_on():
    df = pd.read_csv(test_file)
    predicate = df["Year2"].str.slice(0, 4).astype(int) < 2028
    early, late = split_on(df, predicate)
    assert len(early) == 8
    assert late["Value"].sum() == 116
