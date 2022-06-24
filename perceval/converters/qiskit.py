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

from perceval import Circuit, P
from perceval.algorithm.norm import *
from perceval.algorithm.optimize import optimize
import perceval.lib.qiskit as lib_qiskit
import qiskit

u20a = Circuit(2) // (0, lib_qiskit.PS(P("phi1")))
u20b = Circuit(2) // (1, lib_qiskit.PS(P("phi2")))
u20ab = Circuit(2) // (0, lib_qiskit.PS(P("phi1"))) // (1, lib_qiskit.PS(P("phi2")))

u21 = Circuit(2) // lib_qiskit.BS(P("theta"), P("phi"), P("lambda")) // (0, lib_qiskit.PS(P("Phi")))

hcnot_herald = (lib_qiskit.Circuit(8, name="Heralded CNOT")
                .add((0, 1, 2), lib_qiskit.PERM([2, 1, 0]))
                .add((4, 5), lib_qiskit.BS())
                .add((5, 6, 7), lib_qiskit.PERM([2, 1, 0]))
                .add((3, 4), lib_qiskit.BS())
                .add((2, 3), lib_qiskit.BS(theta=0.23, phi=np.pi))
                .add((4, 5), lib_qiskit.BS(theta=0.23))
                .add((3, 4), lib_qiskit.BS())
                .add((5, 6), lib_qiskit.PERM([1, 0]))
                .add((1, 2), lib_qiskit.PERM([1, 0]))
                .add((2, 3), lib_qiskit.BS(theta=0.76))
                .add((4, 5), lib_qiskit.BS(theta=0.76, phi=np.pi))
                .add((5, 6, 7), lib_qiskit.PERM([1, 2, 0]))
                .add((4, 5), lib_qiskit.BS())
                .add((0, 1, 2), lib_qiskit.PERM([2, 0, 1])))

hcnot_ralph = (lib_qiskit.Circuit(6, name="CNOT - postprocess")
               .add((0, 1), lib_qiskit.BS(1.9106332362,  phi=np.pi))
               .add((3, 4), lib_qiskit.BS())
               .add((2, 3), lib_qiskit.BS(1.9106332362, phi=np.pi))
               .add((4, 5), lib_qiskit.BS(1.9106332362))
               .add((3, 4), lib_qiskit.BS()))


def converter(qc: qiskit.QuantumCircuit, heralded: bool=True) -> Circuit:
    r"""convert a qiskit circuit into a predefined processor

    :param c:
    :return:
    """

    n_cnot = 0
    for instruction in qc.data:
        if instruction[0].name == "cx":
            n_cnot += 1

    nmode = qc.qregs[0].size * 2 + (heralded and 4 or 2) * n_cnot
    pc = Circuit(nmode)
    idx_herald = qc.qregs[0].size * 2

    for instruction in qc.data:
        if isinstance(instruction[0], qiskit.circuit.barrier.Barrier):
            continue
        assert isinstance(instruction[0], qiskit.circuit.gate.Gate), "cannot convert (%s)" % instruction[0]
        if instruction[0].num_qubits == 1:
            u = instruction[0].to_matrix()
            if abs(u[1,0])+abs(u[1,0]) < 1e-4:
                if abs(u[0,0]-1) < 1e-4:
                    if abs(u[1,1]-1) < 1e-4:
                        continue
                    uphi = u20b
                else:
                    if abs(u[1,1]-1) < 1e-4:
                        uphi = u20a
                    else:
                        uphi = u20ab
                optimize(uphi, u, frobenius, sign=-1)
                ins = uphi.copy()
                uphi.reset_parameters()
            else:
                optimize(u21, u, frobenius, sign=-1)
                ins = u21.copy()
                if abs(float(ins._components[1][1]["phi"])) < 1e-4:
                    ins._components = ins._components[:1]
                u21.reset_parameters()
            ins._name = instruction[0].name
            pc.add(instruction[1][0].index*2, ins, merge=False)
        else:
            c_idx = instruction[1][0].index * 2
            c_data = instruction[1][1].index * 2
            if instruction[0].name == "swap":
                pc.add(c_idx, lib_qiskit.PERM([c_idx-c_data, c_idx+1-c_data, 0, 1]))
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
                    pc.add(c_idx, lib_qiskit.PERM(perm))
                    pc.add(c_idx, hcnot_herald, merge=False)
                    pc.add(c_idx, lib_qiskit.PERM(iperm))
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
                    pc.add(c_idx, lib_qiskit.PERM(perm))
                    pc.add(c_idx, hcnot_ralph, merge=False)
                    pc.add(c_idx, lib_qiskit.PERM(iperm))
                    idx_herald += 2

    return pc
