import perceval as pcvl
import perceval.lib.phys as phys

import numpy as np


def test_determinant_base():
    c = phys.BS()
    assert abs(c.U.det().simplify()) == 1


def test_determinant_generic():
    c = phys.BS(theta=pcvl.P("θ"), phi_a=pcvl.P("phi_a"), phi_b=pcvl.P("phi_b"), phi_d=pcvl.P("phi_d"))
    assert abs(c.U.det().simplify()) == 1


def test_determinant_1():
    c = phys.BS(theta=pcvl.P("θ"), phi_a=np.pi/2, phi_b=np.pi/2, phi_d=0)
    assert abs(c.U.det().simplify()) == 1


def test_determinant_2():
    c = phys.BS(theta=pcvl.P("θ"), phi_a=np.pi/2, phi_b=np.pi/2, phi_d=np.pi/2)
    assert abs(c.U.det().simplify()) == 1


def test_determinant_3():
    c = phys.BS(theta=pcvl.P("θ"), phi_a=0, phi_b=0, phi_d=0)
    assert abs(c.U.det().simplify()) == 1
