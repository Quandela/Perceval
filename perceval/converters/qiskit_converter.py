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

from perceval.components import Port, Circuit, Processor, Source
from perceval.utils import P, BasicState, Encoding
from perceval.utils.algorithms.optimize import optimize
from perceval.utils.algorithms.norm import frobenius
import perceval.components.unitary_components as comp


min_precision_gate = 1e-4


class QiskitConverter:
    r"""Qiskit quantum circuit to perceval processor converter.

    :param catalog: a component library to use for the conversion. It must contain CNOT gates.
    :param backend_name: backend name used in the converted processor
    :param source: the source used as input for the converted processor (default perfect source).
    """
    def __init__(self, catalog, backend_name: str = "SLOS", source: Source = Source()):
        self._source = source
        self._heralded_cnot_builder = catalog["heralded cnot"]
        self._heralded_cz_builder = catalog["heralded cz"]
        self._postprocessed_cnot_builder = catalog["postprocessed cnot"]
        self._generic_2mode_builder = catalog["generic 2 mode circuit"]
        self._lower_phase_component = Circuit(2) // (0, comp.PS(P("phi2")))
        self._upper_phase_component = Circuit(2) // (1, comp.PS(P("phi1")))
        self._two_phase_component = Circuit(2) // (0, comp.PS(P("phi1"))) // (1, comp.PS(P("phi2")))
        self._backend_name = backend_name

    def convert(self, qc, use_postselection: bool = True) -> Processor:
        r"""Convert a qiskit quantum circuit into a `Processor`.

        :param qc: quantum-based qiskit circuit
        :type qc: qiskit.QuantumCircuit
        :param use_postselection: when True, uses a `postprocessed CNOT` as the last gate. Otherwise, uses only
            `heralded CNOT`
        :return: the converted processor
        """
        import qiskit  # this nested import fixes automatic class reference generation

        # count the number of cnot to use during the conversion, will give us the number of herald to handle
        n_cnot = 0
        for instruction in qc.data:
            if instruction[0].name == "cx":
                n_cnot += 1
        cnot_idx = 0

        n_moi = qc.qregs[0].size * 2  # number of modes of interest
        input_list = [0] * n_moi
        p = Processor(self._backend_name, n_moi, self._source)
        qubit_names = qc.qregs[0].name
        for i in range(qc.qregs[0].size):
            p.add_port(i * 2, Port(Encoding.DUAL_RAIL, f'{qubit_names}{i}'))
            input_list[i * 2] = 1
        default_input_state = BasicState(input_list)

        for instruction in qc.data:
            # barrier has no effect
            if isinstance(instruction[0], qiskit.circuit.barrier.Barrier):
                continue
            # some limitation in the conversion, in particular measure
            assert isinstance(instruction[0], qiskit.circuit.gate.Gate), "cannot convert (%s)" % instruction[0]

            if instruction[0].num_qubits == 1:
                # one mode gate
                ins = self._create_one_qubit_gate(instruction[0].to_matrix())
                ins._name = instruction[0].name
                p.add(instruction[1][0].index * 2, ins.copy())
            else:
                c_idx = instruction[1][0].index * 2
                c_data = instruction[1][1].index * 2
                c_first = min(c_idx, c_data)

                if instruction[0].name == "swap":
                    # c_idx and c_data are consecutive - not necessarily ordered
                    p.add(c_first, comp.PERM([2, 3, 0, 1]))
                elif instruction[0].name == "cz":
                    cz_processor = self._heralded_cz_builder.build()
                    mode_map = {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}
                    p.add(mode_map, cz_processor)
                elif instruction[0].name == "cx":
                    cnot_idx += 1
                    if use_postselection and cnot_idx == n_cnot:
                        cnot_processor = self._postprocessed_cnot_builder.build()
                        mode_map = {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}
                    else:
                        cnot_processor = self._heralded_cnot_builder.build()
                        mode_map = {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}
                    p.add(mode_map, cnot_processor)

                else:
                    raise RuntimeError("Gate not yet supported: %s" % instruction[0].name)
        p.with_input(default_input_state)
        return p

    def _create_one_qubit_gate(self, u):
        if abs(u[1, 0]) + abs(u[0, 1]) < 2 * min_precision_gate:
            # diagonal matrix - we can handle with phases, we consider that gate unitary parameters has
            # limited numeric precision
            if abs(u[0, 0] - 1) < min_precision_gate:
                if abs(u[1, 1] - 1) < min_precision_gate:
                    return None
                ins = self._upper_phase_component.copy()
            else:
                if abs(u[1, 1] - 1) < min_precision_gate:
                    ins = self._lower_phase_component.copy()
                else:
                    ins = self._two_phase_component.copy()
            optimize(ins, u, frobenius, sign=-1)
        else:
            ins = self._generic_2mode_builder.build()
            optimize(ins, u, frobenius, sign=-1)
        return ins
