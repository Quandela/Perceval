from perceval.utils import mlstr
import numpy as np


def test_basic0():
    s = mlstr(0.234)
    assert str(s) == "0.234"


def test_basic1():
    s = mlstr()
    assert str(s) == ""


def test_basic2():
    s = mlstr("string")
    assert str(s) == "string"


def test_iadd0():
    s = mlstr("string")
    s += "123"
    assert str(s) == "string123"


def test_iadd1():
    s = mlstr("M = ")
    s += "|0 1|\n|1 0|"
    assert str(s) == "M = |0 1|\n    |1 0|"


def test_radd():
    assert str(1+mlstr("a\nb")) == "1a\n b"


def test_iadd_inv():
    s = "M = "
    s + mlstr("|0 1|\n|1 0|")


def test_format():
    s = mlstr("%s = 1/%f * %s")
    assert str(s % ("M", np.sqrt(2), "|0 1|\n|1 0|")) == "M = 1/1.414214 * |0 1|\n                 |1 0|"


def test_join():
    s = mlstr(" ").join(["a", "a\nb", "c", "a\ng\nc"])
    assert str(s) == "a a c a\n  b   g\n      c"
