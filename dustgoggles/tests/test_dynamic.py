from dustgoggles.dynamic import Dynamic


def test_dynamic_1():
    testdef = """def f(x: float) -> float:\n    return x + 1"""
    dyn = Dynamic(testdef, optional=True)
    assert dyn(1) == 2
    dyn("j")
    assert (
        str(dyn.errors[0]["exception"])
        == 'can only concatenate str (not "int") to str'
    )