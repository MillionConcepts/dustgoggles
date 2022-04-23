import random

from pytest import importorskip

np = importorskip("numpy")

from dustgoggles.codex.implements import GridPaper
from dustgoggles.test_utils import wrap_piped


def square(x):
    return x ** 2


def grid_insert(func):
    def grid_lookup(gridpaper_title, key):
        return func(GridPaper(gridpaper_title).get(key))

    return grid_lookup


def test_gridpaper_1():
    address = random.randint(100000, 2000000)
    remote_square = wrap_piped(grid_insert(square))
    gridpaper = GridPaper(f"test_gridpaper_{address}", create=True)
    gridpaper["cat"] = np.diag([2, 2, 2])
    gridpaper["dog"] = np.diag([3, 3, 3])
    product = np.dot(
        remote_square(f"test_gridpaper_{address}", "cat"),
        remote_square(f"test_gridpaper_{address}", "dog")
    )
    assert np.all(product == np.diag([36, 36, 36]))