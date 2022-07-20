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

from perceval import PredefinedCircuit, Circuit, Processor, P, Source, BasicState
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


class QiskitConverter:

    def __init__(self, library, source: Source = None):
        r"""Initialize qiskit to perceval circuit converter.

        :param library: a component library to use for the conversion
        :param source: the source used as input. Default is a perfect source.
        """
        if source is None:
            source = Source()
        self._source = source
        self._lib = library

    def convert(self, qc: qiskit.QuantumCircuit, heralded: bool = None) -> Processor:
        r"""Convert a qiskit circuit into a perceval.Processor.

        :param qc: quantum-based qiskit circuit
        :type qc:  qiskit.QuantumCircuit
        :param heralded: do we use `heralded_cnot` or `post_processed_cnot`, if not set, all cnot but the
            last are heralded
        :return: a processor
        """

        # count the number of cnot to use during the conversion, will give us the number of herald to handle
        n_cnot = 0
        for instruction in qc.data:
            if instruction[0].name == "cx":
                n_cnot += 1

        cnot_component_heralded = self._lib.catalog["heralded_cnot"]
        cnot_component_postprocessed = self._lib.catalog["post_processed_cnot"]
        generic_2mode_component = self._lib.catalog["generic_2mode"]
        lower_phase_component = PredefinedCircuit(self._lib.Circuit(2) // (0, self._lib.PS(P("phi2"))))
        upper_phase_component = PredefinedCircuit(self._lib.Circuit(2) // (1, self._lib.PS(P("phi1"))))
        two_phase_component = PredefinedCircuit(
            self._lib.Circuit(2) // (0, self._lib.PS(P("phi1"))) // (1, self._lib.PS(P("phi2"))))

        qubit_names = qc.qregs[0].name
        sources = {}
        port_names = {}
        for i in range(qc.qregs[0].size):
            sources[2*i] = self._source
            port_names[2*i] = "%s_%s:%d" % (qubit_names, i, 0)
            port_names[2*i+1] = "%s_%s:%d" % (qubit_names, i, 1)

        post_select_fn = lambda s: True
        global_heralds = {}

        n_modes = qc.qregs[0].size * 2
        if n_cnot:
            if heralded is True:
                n_modes += len(cnot_component_heralded.heralds) * n_cnot
            elif heralded is False:
                n_modes += len(cnot_component_postprocessed.heralds) * n_cnot
            else:  # self._heralded is None
                n_modes += len(cnot_component_heralded.heralds) * (n_cnot - 1) \
                           + len(cnot_component_postprocessed.heralds)
        cnot_idx = 0
        pc = Circuit(n_modes)
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
                if abs(u[1, 0]) + abs(u[0, 1]) < 2 * min_precision_gate:
                    # diagonal matrix - we can handle with phases, we consider that gate unitary parameters has
                    # limited numeric precision
                    if abs(u[0, 0] - 1) < min_precision_gate:
                        if abs(u[1, 1] - 1) < min_precision_gate:
                            continue
                        ins = upper_phase_component.circuit
                    else:
                        if abs(u[1, 1] - 1) < min_precision_gate:
                            ins = lower_phase_component.circuit
                        else:
                            ins = two_phase_component.circuit
                    optimize(ins, u, frobenius, sign=-1)
                else:
                    ins = generic_2mode_component.circuit
                    optimize(ins, u, frobenius, sign=-1)
                ins._name = instruction[0].name
                pc.add(instruction[1][0].index * 2, ins.copy(), merge=False)
            else:
                c_idx = instruction[1][0].index * 2
                c_data = instruction[1][1].index * 2
                c_first = min(c_idx, c_data)
                if instruction[0].name == "swap":
                    # c_idx and c_data are consecutive - not necessarily ordered
                    pc.add(c_first, self._lib.PERM([2, 3, 0, 1]))
                else:
                    cnot_idx += 1
                    if heralded is False or (heralded is None and cnot_idx == n_cnot):
                        cnot_component = cnot_component_postprocessed
                    else:
                        cnot_component = cnot_component_heralded
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
                        max_port = idx_herald + len(heralds)
                    else:
                        max_port = max(c_idx, c_data) + 1
                    cnot_component_instance = cnot_component.circuit
                    c_last = c_first + cnot_component_instance.m
                    real_port = 0
                    # list all port permutations
                    inv_perm = []
                    perm = list(range(max_port - min_port))
                    # used ports
                    used_ports = list(range(max_port - len(heralds) - min_port))
                    # plug-in all necessary ports entering the component
                    for p_idx in range(cnot_component_instance.m):
                        if p_idx in heralds:
                            inv_perm.append(idx_herald - c_first)
                            perm[idx_herald - c_first] = p_idx
                            if heralds[p_idx]:
                                sources[idx_herald] = self._source
                            idx_herald += 1
                        else:
                            if real_port < 2:
                                # c_idx
                                if c_idx < c_data:
                                    inv_perm.append(real_port)
                                else:
                                    inv_perm.append(c_idx - c_data + real_port)
                            else:
                                # c_data
                                if c_idx < c_data:
                                    inv_perm.append(c_data - c_idx + real_port - 2)
                                else:
                                    inv_perm.append(real_port - 2)
                            perm[inv_perm[-1]] = p_idx
                            if inv_perm[-1] <= len(used_ports):
                                used_ports[inv_perm[-1]] = None
                            real_port += 1

                    for p_idx in used_ports:
                        if p_idx is not None:
                            inv_perm.append(p_idx)
                            perm[p_idx] = len(inv_perm) - 1

                    pc.add(c_first, self._lib.PERM(perm))
                    pc.add(c_first, cnot_component_instance, merge=False)
                    if heralds:
                        for k, v in heralds.items():
                            global_heralds[perm.index(k)+c_first] = v
                    if cnot_component.has_post_select:
                        post_select_fn = lambda s, curr_post_select=post_select_fn: curr_post_select(s) and\
                                                                    cnot_component.post_select(
                                                                        BasicState([s[perm.index(ii)+c_first]
                                                                                    for ii in range(cnot_component_instance.m)]))

                    pc.add(c_first, self._lib.PERM(inv_perm))

        p = Processor(sources, pc, post_select_fn=post_select_fn, heralds=global_heralds)
        p.set_port_names(port_names, port_names)
        return p
