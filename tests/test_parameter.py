from perceval import Parameter

import sympy as sp


def test_definition():
    p = Parameter("alpha", 0)
    assert isinstance(p.spv, sp.Number)
    assert p.spv == 0
    assert p.defined and float(p) == 0


def test_variable():
    p = Parameter("alpha")
    assert isinstance(p.spv, sp.Expr)
    assert not p.defined


def test_set_variable():
    p = Parameter("alpha")
    p.set_value(0.5)
    assert isinstance(p.spv, sp.Number)
    assert p.defined
    assert float(p) == 0.5


def test_fixed_0():
    p = Parameter("alpha", 2)
    assert p.defined
    try:
        p.set_value(1)
    except RuntimeError:
        pass
    else:
        raise Exception("Cannot set a fixed parameter")


def test_fixed_1():
    p = Parameter("alpha")
    assert not p.fixed
    assert not p.defined
    p.set_value(1)
    assert not p.fixed
    assert p.defined


def test_basicconv():
    p = Parameter("R", 1/3)
    assert p._value == sp.S(1)/3
