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

from perceval import PredefinedCircuit, Circuit, Processor, P, Source, Matrix
from perceval.algorithm.norm import *
from perceval.algorithm.optimize import optimize

import qiskit

# TODO: add typing on library

min_precision_gate = 1e-4

def _swap(perm, port_a, port_b):
    if port_a != port_b:
        c = perm[port_a]
        perm[port_a] = perm[port_b]
        perm[port_b] = c

def to_perceval(qc: qiskit.QuantumCircuit, library, heralded : bool=True) -> Processor:
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
        # barrier has no effect
        if isinstance(instruction[0], qiskit.circuit.barrier.Barrier):
            continue
        # some limitation in the conversion, in particular measure
        assert isinstance(instruction[0], qiskit.circuit.gate.Gate), "cannot convert (%s)" % instruction[0]

        if instruction[0].num_qubits == 1:
            # one mode gate
            u = instruction[0].to_matrix()
            if abs(u[1,0])+abs(u[0,1]) < 2 * min_precision_gate:
                # diagonal matrix - we can handle with phases, we consider that gate unitary parameters has
                # limited numeric precision
                if abs(u[0,0]-1) < min_precision_gate:
                    if abs(u[1,1]-1) < min_precision_gate:
                        continue
                    ins = upper_phase_component.circuit
                else:
                    if abs(u[1,1]-1) < min_precision_gate:
                        ins = lower_phase_component.circuit
                    else:
                        ins = two_phase_component.circuit
                optimize(ins, u, frobenius, sign=-1)
            else:
                ins = generic_2mode_component.circuit
                optimize(ins, u, frobenius, sign=-1)
            ins._name = instruction[0].name
            pc.add(instruction[1][0].index*2, ins.copy(), merge=False)
        else:
            c_idx = instruction[1][0].index * 2
            c_data = instruction[1][1].index * 2
            c_first = min(c_idx, c_data)
            if instruction[0].name == "swap":
                # c_idx and c_data are consecutive - not necessarily ordered
                pc.add(c_first, library.PERM([2, 3, 0, 1]))
            else:
                assert instruction[0].name == "cx", "gate not yet supported: %s" % instruction[0].name
                # convert cx to cnot - adding potential heralds
                heralds = cnot_component.heralds
                # need a global permutation from c_idx to c_data if no heralds, or c_idx to last herald otherwise
                # the permutation will:
                # - move the herald to new herald line we create
                # - move c_idx to the first 2 ports of the cnot component
                # - move c_idx to the second 2 ports of the cnot component
                # - move intermediate lines and all lines on the way of the component below
                min_port = c_first
                if heralds:
                    first_herald = idx_herald - min_port
                    max_port = idx_herald + len(heralds)
                else:
                    first_herald = None
                    max_port = max(c_idx, c_data) + 1
                cnot_component_instance = cnot_component.circuit
                real_port = 0
                # list all port permutations
                inv_perm = []
                perm = list(range(max_port-min_port))
                # used ports
                used_ports = list(range(max_port-len(heralds)-min_port))
                # plug-in all necessary ports entering in the component
                for p_idx in range(cnot_component_instance.m):
                    if p_idx in heralds:
                        inv_perm.append(idx_herald-c_first)
                        perm[idx_herald-c_first] = p_idx
                        if heralds[p_idx]:
                            sources[idx_herald] = s
                        idx_herald += 1
                    else:
                        if real_port < 2:
                            # c_idx
                            if c_idx < c_data:
                                inv_perm.append(real_port)
                            else:
                                inv_perm.append(c_idx-c_data+real_port)
                        else:
                            # c_data
                            if c_idx < c_data:
                                inv_perm.append(c_data-c_idx+real_port-2)
                            else:
                                inv_perm.append(real_port-2)
                        perm[inv_perm[-1]] = p_idx
                        if inv_perm[-1] <= len(used_ports):
                            used_ports[inv_perm[-1]] = None
                        real_port += 1

                for p_idx in used_ports:
                    if p_idx is not None:
                        inv_perm.append(p_idx)
                        perm[p_idx] = len(inv_perm)-1

                pc.add(c_first, library.PERM(perm))
                pc.add(c_first, cnot_component_instance, merge=False)
                pc.add(c_first, library.PERM(inv_perm))

    return Processor(sources, pc)
