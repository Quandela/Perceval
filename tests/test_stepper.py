import numpy as np
import perceval as pcvl
import perceval.lib.phys as phys
import perceval.lib.symb as symb

import pytest

from test_simulators import check_output


def test_minimal():
    # default simulator backend
    simulator_backend = pcvl.BackendFactory().get_backend("Stepper")
    # simulator directly initialized on circuit
    s = simulator_backend(phys.BS())
    check_output(s, pcvl.AnnotatedBasicState([1, 1]), {pcvl.BasicState("|1,0>"): 0,
                                                       pcvl.BasicState("|0,1>"): 0,
                                                       pcvl.BasicState("|0,2>"): 0.5,
                                                       pcvl.BasicState("|2,0>"): 0.5})


def test_c3():
    for backend in ["Stepper", "Naive", "SLOS"]:
        # default simulator backend
        simulator_backend = pcvl.BackendFactory().get_backend(backend)
        # simulator directly initialized on circuit
        circuit = pcvl.Circuit(3)
        circuit.add((0, 1), phys.BS())
        circuit.add((1,), phys.PS(np.pi/4))
        circuit.add((1, 2), phys.BS())
        pcvl.pdisplay(circuit.U)
        s = simulator_backend(circuit)
        check_output(s, pcvl.AnnotatedBasicState([0, 1, 1]), {pcvl.BasicState("|0,1,1>"): 0,
                                                              pcvl.BasicState("|1,1,0>"): 0.25,
                                                              pcvl.BasicState("|1,0,1>"): 0.25,
                                                              pcvl.BasicState("|2,0,0>"): 0,
                                                              pcvl.BasicState("|0,2,0>"): 0.25,
                                                              pcvl.BasicState("|0,0,2>"): 0.25,
                                                              })

@pytest.mark.skip(reason="need to fix delay implementation")
def test_timedelay():
    dt = pcvl.Parameter("Δt")

    c = pcvl.Circuit(2)
    c //= symb.BS()
    c //= ((1), symb.DT(dt))
    c //= symb.BS()

    st0 = pcvl.AnnotatedBasicState([1, 0], pcvl.TimeDescription(length=0.2))
    stv = pcvl.StateVector(period=2)
    stv[st0] = 1
    backend = pcvl.BackendFactory().get_backend("Stepper")

    sim = backend(c)

    dt.set_value(0)
    assert sim.prob(stv, pcvl.BasicState([2, 0]))+sim.prob(stv, pcvl.BasicState([0, 2])) == 0

    dt.set_value(2e-9)
    assert pytest.approx(sim.prob(stv, pcvl.BasicState([2, 0]))+sim.prob(stv, pcvl.BasicState([0, 2]))) == 0.5


def test_basic_interference():
    simulator_backend = pcvl.BackendFactory().get_backend("Stepper")
    c = phys.BS()
    sim = simulator_backend(c, use_symbolic=False)
    assert pytest.approx(sim.prob(pcvl.BasicState([1, 1]), pcvl.BasicState([2, 0]))) == 0.5
