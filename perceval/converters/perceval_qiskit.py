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

from perceval import PredefinedCircuit, Circuit, Processor, P, Source
from perceval.algorithm.norm import *
from perceval.algorithm.optimize import optimize

import qiskit

# TODO: add typing on library

def converter(qc: qiskit.QuantumCircuit, library, heralded : bool=True) -> Processor:
    r"""convert a qiskit circuit into a predefined processor

    :param qc: quantum-based qiskit circuit
    :type qc:  qiskit.QuantumCircuit
    :param library: a component library to use for the conversion
    :param heralded: do we use `heralded_cnot` or `post_processed_cnot`
    :return: a processor
    """

    # count the number of cnot to use during the conversion, will give us the number of herald to handle
    n_cnot = 0
    for instruction in qc.data:
        if instruction[0].name == "cx":
            n_cnot += 1

    cnot_component = heralded and library.catalog["heralded_cnot"] or library.catalog["post_processed_cnot"]
    generic_2mode_component = library.catalog["generic_2mode"]
    lower_phase_component = PredefinedCircuit(library.Circuit(2) // (0, library.PS(P("phi2"))))
    upper_phase_component = PredefinedCircuit(library.Circuit(2) // (1, library.PS(P("phi1"))))
    two_phase_component = PredefinedCircuit(library.Circuit(2) // (0, library.PS(P("phi1"))) // (1, library.PS(P("phi2"))))

    herald_per_cnot = len(cnot_component.heralds)

    # define the sources - let us suppose a single perfect source
    s = Source()
    sources = {i*2: s for i in range(qc.qregs[0].size)}

    nmode = qc.qregs[0].size * 2 + herald_per_cnot * n_cnot
    pc = Circuit(nmode)
    idx_herald = qc.qregs[0].size * 2

    for instruction in qc.data:
        if isinstance(instruction[0], qiskit.circuit.barrier.Barrier):
            continue
        assert isinstance(instruction[0], qiskit.circuit.gate.Gate), "cannot convert (%s)" % instruction[0]
        if instruction[0].num_qubits == 1:
            u = instruction[0].to_matrix()
            if abs(u[1,0])+abs(u[0,1]) < 1e-4:
                # diagonal matrix - we can handle with phases, 1e-4 is arbitrary here
                if abs(u[0,0]-1) < 1e-4:
                    if abs(u[1,1]-1) < 1e-4:
                        continue
                    ins = upper_phase_component.circuit
                else:
                    if abs(u[1,1]-1) < 1e-4:
                        ins = lower_phase_component.circuit
                    else:
                        ins = two_phase_component.circuit
                optimize(ins, u, frobenius, sign=-1)
            else:
                ins = generic_2mode_component.circuit
                optimize(ins, u, frobenius, sign=-1)
            ins._name = instruction[0].name
            pc.add(instruction[1][0].index*2, ins, merge=False)
        else:
            c_idx = instruction[1][0].index * 2
            c_data = instruction[1][1].index * 2
            if instruction[0].name == "swap":
                pc.add(c_idx, library.PERM([c_idx-c_data, c_idx+1-c_data, 0, 1]))
            else:
                assert instruction[0].name == "cx", "gate not yet supported: %s" % instruction[0].name
                if heralded:
                    # we need to send:
                    #   - idx_herald to c_idx
                    #   - c_idx to c_idx+2
                    #   - c_data to c_idx+4
                    #   - idx_herald+2 to c_idx+6
                    perm = [2, 3]
                    iperm = [idx_herald-c_idx, idx_herald-c_idx+1]
                    iperm += [0, 1, c_data-c_idx, c_data+1-c_idx]
                    offset = 8
                    ioffset = 2
                    if c_data > c_idx + 2:
                        for p in range(c_idx+2, c_data):
                            perm += [offset]
                            iperm += [ioffset]
                            offset += 1
                            ioffset += 1
                    perm += [4, 5]
                    iperm += [idx_herald+2-c_idx, idx_herald-c_idx+3]
                    ioffset += 2
                    for p in range(c_data+2, idx_herald):
                        perm += [offset]
                        iperm += [ioffset]
                        offset += 1
                        ioffset += 1
                    perm += [0, 1]
                    perm += [6, 7]
                    pc.add(c_idx, library.PERM(perm))
                    pc.add(c_idx, cnot_component, merge=False)
                    pc.add(c_idx, library.PERM(iperm))
                    idx_herald += 4
                else:
                    # we need to send:
                    #   - idx_herald to c_idx
                    #   - c_idx to c_idx+1
                    #   - c_data to c_idx+3
                    #   - idx_herald+1 to c_idx+5
                    perm = [1, 2]
                    iperm = [idx_herald-c_idx]
                    iperm += [0, 1, c_data-c_idx, c_data+1-c_idx]
                    offset = 6
                    ioffset = 2
                    if c_data > c_idx + 2:
                        for p in range(c_idx+2, c_data):
                            perm += [offset]
                            iperm += [ioffset]
                            offset += 1
                            ioffset += 1
                    perm += [3, 4]
                    iperm += [idx_herald+1-c_idx]
                    ioffset += 2
                    for p in range(c_data+2, idx_herald):
                        perm += [offset]
                        iperm += [ioffset]
                        offset += 1
                        ioffset += 1
                    perm += [0]
                    perm += [5]
                    pc.add(c_idx, library.PERM(perm))
                    pc.add(c_idx, cnot_component, merge=False)
                    pc.add(c_idx, library.PERM(iperm))
                    idx_herald += 2

    return Processor(sources, pc)
