import json
from pathlib import Path

import pytest
pd = pytest.importorskip('pandas')

test_file = Path(
    Path(__file__).parent.parent, 'test_utils', 'data', 'm20_waypoints.json'
)


# def test_extract_constants():
#     with test_file.open() as file:
#         df = pd.DataFrame(json.load(file)['features'])
