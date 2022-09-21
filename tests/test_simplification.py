import perceval as pcvl
from perceval.utils.algorithms.simplification import *
from perceval.components import base_components as comp


def PS_testing(circ, display):
    c2 = simplify(circ, display=display)

    real = []
    for r, c in c2:
        if isinstance(c, comp.PS):
            phi = c.get_variables()["phi"]
            real.append((r[0], phi))

    return real


def test_PS_simp():
    phi = pcvl.P("phi")

    c = (Circuit(3)
         .add(0, comp.PS(np.pi))
         .add(0, comp.PERM([2, 1, 0]))
         .add(0, comp.SimpleBS())
         .add(2, comp.PS(phi))
         .add(2, comp.PS(np.pi))
         .add(0, comp.PS(np.pi / 2)))

    expected = [(0, 2 * np.pi), (2, "phi"), (0, np.pi /2)]
    real = PS_testing(c, True)

    assert real == expected, "PS simplification with display = True not passed"

    expected = [(2, "phi"), (0, np.pi /2)]
    real = PS_testing(c, False)

    assert real == expected, "PS simplification with display = False not passed"


def PERM_testing(circ):
    real = []

    c2 = simplify(circ)

    for r, c in c2:
        if isinstance(c, comp.PERM):
            real.append((r[0], c.perm_vector))
        elif isinstance(c, comp.SimpleBS):
            real.append((r[0],))

    return real


def test_perm_simp():
    circ = (Circuit(3)
            .add(0, comp.PERM([0, 2, 1])))

    expected = [(1, [1, 0])]


    real = PERM_testing(circ)

    assert real == expected, "PERM reduction is wrong"

    circ = (Circuit(3)
            .add(0, comp.PERM([0, 2, 1]))
            .add(0, comp.PERM([1, 2, 0])))

    expected = [(0, [1, 0])]
    real = PERM_testing(circ)

    assert real == expected, "PERM reduction is wrong"

    c = (Circuit(3)
         .add(0, comp.PERM([2, 0, 1]))
         .add(0, comp.SimpleBS())
         .add(0, comp.PERM([1, 2, 0])))

    expected = [(1,)]
    real = PERM_testing(c)

    assert real == expected, "PERM simplification moves components wrongly"
