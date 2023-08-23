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

import random

import perceval as pcvl
from perceval.backends import SLOSBackend, NaiveBackend
from perceval.simulators import Stepper
from perceval.components import BS, PS
from perceval.utils import BasicState

# definition of the circuit
C = pcvl.Circuit(2)
C.add((0, 1), BS())
C.add(1, PS(1))
C.add((0, 1), BS())

N = 100


def get_sample_from_statevector(sv):
    p = random.random()
    state = None
    for state, pa in sv.items():
        proba = abs(pa)**2
        if p > proba:
            p -= proba
            continue
        break
    return state


def run_stepper():
    samples = []
    stepper = Stepper(SLOSBackend())
    stepper.set_circuit(C)
    for i in range(N):
        sv = pcvl.StateVector(pcvl.BasicState([1, 0]))
        for r, c in C:
            sv = stepper.apply(sv, r, c)
        samples.append(get_sample_from_statevector(sv))
    return samples


def run_direct():
    bs = C._components[0][1]
    sim_bs = NaiveBackend()
    sim_bs.set_circuit(bs)
    ps = C._components[1][1]
    sim_ps = NaiveBackend()
    sim_ps.set_circuit(ps)
    samples = []
    bs10 = BasicState([1,0])
    bs01 = BasicState([0,1])
    for i in range(N):
        # apply first bs
        sim_bs.set_input_state(bs10)
        sv_a0 = sim_bs.prob_amplitude(bs10)
        sv_a1 = sim_bs.prob_amplitude(bs01)
        # apply ps
        sv_b0 = sv_a0
        sim_ps.set_input_state(BasicState([1]))
        sv_b1 = sv_a1*sim_ps.prob_amplitude(BasicState([1]))
        # apply second bs
        sv_c0 = sv_b0*sim_bs.prob_amplitude(bs10)
        sv_c1 = sv_b0*sim_bs.prob_amplitude(bs01)
        sim_bs.set_input_state(bs01)
        sv_c0 += sv_b1*sim_bs.prob_amplitude(bs10)
        sv_c1 += sv_b1*sim_bs.prob_amplitude(bs01)
        # sampling from there
        samples.append(bs10 if random.random() > abs(sv_c0)**2 else bs01)
    return samples


def test_stepper(benchmark):
    benchmark(run_stepper)


def test_stepper_comp_direct(benchmark):
    benchmark(run_direct)
