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
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from perceval.simulators import SimulatorFactory, Simulator, DelaySimulator, LossSimulator, PolarizationSimulator
from perceval.components import BS, PBS, Unitary, PS, TD, LC, Processor
from perceval.backends._slos import SLOSBackend
from perceval.backends._naive import NaiveBackend
from perceval.utils import BasicState

import numpy as np


def test_create_simulator_from_circuit():
    c = BS()
    simu = SimulatorFactory.build(c)
    assert isinstance(simu, Simulator)
    assert isinstance(simu._backend, SLOSBackend)  # Default backend is SLOS
    assert simu._backend._circuit == c

    simu = SimulatorFactory.build(c, SLOSBackend())
    assert isinstance(simu, Simulator)
    assert isinstance(simu._backend, SLOSBackend)

    simu = SimulatorFactory.build(c, "SLOS")
    assert isinstance(simu, Simulator)
    assert isinstance(simu._backend, SLOSBackend)

    simu = SimulatorFactory.build(c, "Naive")
    assert isinstance(simu, Simulator)
    assert isinstance(simu._backend, NaiveBackend)


def test_create_simulator_from_polarized_circuit():
    c2 = PBS()
    simu = SimulatorFactory.build(c2)
    assert isinstance(simu, PolarizationSimulator)
    assert isinstance(simu._simulator, Simulator)
    assert isinstance(simu._simulator._backend, SLOSBackend)
    # PolarizationSimulator prepares _upol and the circuit is set only when a polarized input is set
    assert np.allclose(simu._upol, c2.compute_unitary())
    simu.probs(BasicState('|{P:H},0>'))
    assert isinstance(simu._simulator._backend._circuit, Unitary)  # The resulting circuit is set as a Unitary


def test_create_simulator_from_components():
    cp_list_1 = [((0,1), BS()),
                 ((1,), PS(phi=2)),
                 ((0,1), BS())]
    simu = SimulatorFactory.build(cp_list_1, "SLOS")
    assert isinstance(simu, Simulator)
    assert isinstance(simu._backend, SLOSBackend)

    cp_list_2 = [((0,1), BS()),
                 ((1,), TD(dt=2)),
                 ((0,1), BS())]
    simu = SimulatorFactory.build(cp_list_2, "Naive")
    assert isinstance(simu, DelaySimulator)
    assert isinstance(simu._simulator, Simulator)
    assert isinstance(simu._simulator._backend, NaiveBackend)

    cp_list_3 = [((0,1), BS()),
                 ((1,), LC(loss=0.2)),
                 ((0,1), BS())]
    simu = SimulatorFactory.build(cp_list_3)
    assert isinstance(simu, LossSimulator)
    assert isinstance(simu._simulator, Simulator)
    assert isinstance(simu._simulator._backend, SLOSBackend)


def test_create_simulator_from_complex_processor():
    p = Processor("SLOS", 2)
    p.add(0, BS())
    p.add(0, TD(dt=1))  # Triggers the use of DelaySimulator
    p.add(0, PS(phi=0.5))
    p.add(1, LC(loss=0.1))  # Triggers the use of LossSimulator
    p.add(0, PBS())  # Triggers the use of PolarizationSimulator

    simu = SimulatorFactory.build(p)
    assert isinstance(simu, LossSimulator)
    assert isinstance(simu._simulator, DelaySimulator)
    assert isinstance(simu._simulator._simulator, PolarizationSimulator)
    assert isinstance(simu._simulator._simulator._simulator, Simulator)
