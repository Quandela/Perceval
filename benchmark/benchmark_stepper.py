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

import random

import perceval as pcvl
from perceval.components.base_components import BS, PS

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
    stepper = pcvl.BackendFactory().get_backend("Stepper")(C)
    for i in range(N):
        sv = pcvl.StateVector(pcvl.BasicState([1, 0]))
        for r, c in C:
            sv = stepper.apply(sv, r, c)
        samples.append(get_sample_from_statevector(sv))
    return samples


def run_direct():
    u_bs = C._components[0][1].compute_unitary(use_symbolic=False)
    sim_bs = pcvl.BackendFactory().get_backend("Naive")(u_bs)
    u_ps = C._components[1][1].compute_unitary(use_symbolic=False)
    sim_ps = pcvl.BackendFactory().get_backend("Naive")(u_ps)
    samples = []
    for i in range(N):
        # apply first bs
        sv_a0 = sim_bs.probampli(pcvl.BasicState([1,0]), pcvl.BasicState([1,0]))
        sv_a1 = sim_bs.probampli(pcvl.BasicState([1,0]), pcvl.BasicState([0,1]))
        # apply ps
        sv_b0 = sv_a0
        sv_b1 = sv_a1*sim_ps.probampli(pcvl.BasicState([1]), pcvl.BasicState([1]))
        # apply second bs
        sv_c0 = sv_b0*sim_bs.probampli(pcvl.BasicState([1,0]), pcvl.BasicState([1,0]))
        sv_c1 = sv_b0*sim_bs.probampli(pcvl.BasicState([1,0]), pcvl.BasicState([0,1]))
        sv_c0 += sv_b1*sim_bs.probampli(pcvl.BasicState([0,1]), pcvl.BasicState([1,0]))
        sv_c1 += sv_b1*sim_bs.probampli(pcvl.BasicState([0,1]), pcvl.BasicState([0,1]))
        # sampling from there
        samples.append(random.random()>abs(sv_c0)**2 and pcvl.BasicState([1,0]) or pcvl.BasicState([0,1]))
    return samples


def run_naive():
    samples = []
    sim_naive = pcvl.BackendFactory().get_backend("Naive")(C.compute_unitary(use_symbolic=False))
    for i in range(N):
        samples.append(sim_naive.sample(pcvl.BasicState([1,0])))
    return samples


def test_stepper(benchmark):
    benchmark(run_stepper)


def test_stepper_comp_naive(benchmark):
    benchmark(run_naive)


def test_stepper_comp_direct(benchmark):
    benchmark(run_direct)
