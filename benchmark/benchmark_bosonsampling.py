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

import perceval as pcvl
import perceval.lib.symb as symb
import numpy as np

def get_matrix_n(n):
    def _gen_mzi(i: int):
        return symb.BS(R=0.42) // symb.PS(np.pi+i*0.1) // symb.BS(R=0.42) // symb.PS(np.pi/2)
    return pcvl.Circuit.generic_interferometer(n, _gen_mzi)


def run_backend(backend, shots, U, input_state):
    sim = pcvl.BackendFactory().get_backend(backend)(U)
    sim.compile(input_state)
    for i in range(shots):
        sim.sample(input_state)


def test_bosonsampling_clifford_6(benchmark):
    benchmark(run_backend, backend="CliffordClifford2017", shots=100,
              U=get_matrix_n(6), input_state=pcvl.BasicState([1]*6))


def test_bosonsampling_slos_6(benchmark):
    benchmark(run_backend, backend="SLOS", shots=100,
              U=get_matrix_n(6), input_state=pcvl.BasicState([1]*6))


def test_bosonsampling_naive_6(benchmark):
    benchmark(run_backend, backend="Naive", shots=100,
              U=get_matrix_n(6), input_state=pcvl.BasicState([1]*6))


def test_bosonsampling_clifford_8(benchmark):
    benchmark(run_backend, backend="CliffordClifford2017", shots=20,
              U=get_matrix_n(8), input_state=pcvl.BasicState([1]*8))


def test_bosonsampling_slos_8(benchmark):
    benchmark(run_backend, backend="SLOS", shots=20,
              U=get_matrix_n(8), input_state=pcvl.BasicState([1]*8))


def test_bosonsampling_naive_8(benchmark):
    benchmark(run_backend, backend="Naive", shots=20,
              U=get_matrix_n(8), input_state=pcvl.BasicState([1]*8))

# run_backend(backend="SLOS", shots=100,
#               U=get_matrix_n(6), input_state=pcvl.BasicState([1]*6))
