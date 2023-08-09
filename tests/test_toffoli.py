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
import pytest
import perceval as pcvl


def test_expected():
    p = pcvl.components.catalog['toffoli'].build_processor()
    normal_list_states = {}
    for i in range(8):
        state = format(i, '#05b')[2:]
        associated_fock_state = "|"
        for b in state:
            if b == '0':
                associated_fock_state += "1,0,"
            elif b == '1':
                associated_fock_state += "0,1,"
            else:
                raise ValueError()
        associated_fock_state = associated_fock_state[:-1]
        associated_fock_state += ">"
        normal_list_states[pcvl.BasicState(associated_fock_state)] = state
    ca = pcvl.algorithm.Analyzer(p, input_states=normal_list_states, output_states=normal_list_states)
    ca.compute(expected={"000": "000", "001": "001","010": "010", "011": "011","100": "100", "101": "101", "110": "111", "111": "110"})
    pcvl.pdisplay(ca)
    print("performance=%s, fidelity=%.3f%%" % (pcvl.simple_float(ca.performance)[1], ca.fidelity * 100))
