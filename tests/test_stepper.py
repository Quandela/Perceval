# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import perceval as pcvl
import perceval.lib.phys as phys

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


def test_basic_interference():
    simulator_backend = pcvl.BackendFactory().get_backend("Stepper")
    c = phys.BS()
    sim = simulator_backend(c, use_symbolic=False)
    assert pytest.approx(sim.prob(pcvl.BasicState([1, 1]), pcvl.BasicState([2, 0]))) == 0.5
