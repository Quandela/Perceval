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
import perceval.lib.phys as phys


def test_permutation_3():
    circuit =  phys.PERM([2, 0, 1])
    simulator_backend = pcvl.BackendFactory().get_backend("SLOS")
    s_circuit = simulator_backend(circuit)
    ca = pcvl.CircuitAnalyser(s_circuit, input_states=[pcvl.AnnotatedBasicState("|1,0,0>")],
                             output_states = "*")
    assert ca.output_states_list[2] == pcvl.BasicState("|0, 0, 1>")
    assert not((ca.distribution[0]-[0, 0, 1]).any())
    ca = pcvl.CircuitAnalyser(s_circuit, input_states=[pcvl.AnnotatedBasicState("|0,1,0>")],
                             output_states = "*")
    assert ca.output_states_list[0] == pcvl.BasicState("|1, 0, 0>")
    assert not((ca.distribution[0]-[1, 0, 0]).any())
    ca = pcvl.CircuitAnalyser(s_circuit, input_states=[pcvl.AnnotatedBasicState("|0,0,1>")],
                             output_states = "*")
    assert ca.output_states_list[1] == pcvl.BasicState("|0, 1, 0>")
    assert not((ca.distribution[0]-[0, 1, 0]).any())
