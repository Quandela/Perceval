import pytest
import perceval as pcvl
import sympy as sp


def test_format_simple():
    assert pcvl.simple_float(1)[1] == "1"
    assert pcvl.simple_float(-0)[1] == "0"
    assert pcvl.simple_float(0.0000000000001)[1] != "1e-10"
    assert pcvl.simple_float(1.0000000000001)[1] == "1"
    assert pcvl.simple_float(-2/3)[1] == "-2/3"
    assert pcvl.simple_float(-2/3, nsimplify=False)[1] == "-0.666667"
    assert pcvl.simple_float(-2/3, nsimplify=False, precision=1e-7)[1] == "-0.6666667"
    assert pcvl.simple_float(-2/30000, nsimplify=False, precision=1e-7)[1] == "-6.6666667e-5"
    assert pcvl.simple_float(float(-23*sp.pi/19))[1] == "-23*pi/19"


def test_format_complex():
    assert pcvl.simple_complex(1)[1] == "1"
    assert pcvl.simple_complex(-0)[1] == "0"
    assert pcvl.simple_complex(0.0000000000001)[1] != "1e-10"
    assert pcvl.simple_complex(1.0000000000001)[1] == "1"
    assert pcvl.simple_complex(-2j/3)[1] == "-2*I/3"
    assert pcvl.simple_complex(complex(1/sp.sqrt(2)-5j*sp.sqrt(5)/3))[1] == "sqrt(2)/2-5*sqrt(5)*I/3"
    assert pcvl.simple_complex(0.001+1e-15j)[1] == "0.001"
    assert pcvl.simple_complex(0.0001+1e-15j)[1] == "1e-4"
