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

from perceval.components import Processor, Source
from .abstract_converter import AGateConverter


class QiskitConverter(AGateConverter):
    r"""Qiskit quantum circuit to perceval processor converter.

    :param catalog: a component library to use for the conversion. It must contain CNOT gates.
    :param backend_name: backend name used in the converted processor (default SLOS)
    :param source: the source used as input for the converted processor (default perfect source).
    """
    def __init__(self, catalog, backend_name: str = "SLOS", source: Source = Source()):
        super().__init__(catalog, backend_name, source)

    def count_qubits(self, gate_circuit) -> int:
        return gate_circuit.qregs[0].size  # number of qubits

    def convert(self, qc, use_postselection: bool = True) -> Processor:
        r"""Convert a qiskit quantum circuit into a `Processor`.

        :param qc: quantum-based qiskit circuit
        :type qc: qiskit.QuantumCircuit
        :param use_postselection: when True, uses a `postprocessed CNOT` as the last gate. Otherwise, uses only
            `heralded CNOT`
        :return: the converted processor
        """
        import qiskit  # this nested import fixes automatic class reference generation

        n_cnot = 0  # count the number of CNOT gates in circuit - needed to find the num. heralds
        for instruction in qc.data:
            if instruction[0].name == "cx":
                n_cnot += 1

        qubit_names = qc.qregs[0].name
        self._configure_processor(qc, qname=qubit_names)  # empty processor with ports initialized

        for instruction in qc.data:
            # barrier has no effect
            if isinstance(instruction[0], qiskit.circuit.barrier.Barrier):
                continue
            # some limitation in the conversion, in particular measure
            assert isinstance(instruction[0], qiskit.circuit.gate.Gate), "cannot convert (%s)" % instruction[0]

            if instruction[0].num_qubits == 1:
                # one mode gate
                ins = self._create_generic_1_qubit_gate(instruction[0].to_matrix())
                ins._name = instruction[0].name
                self._converted_processor.add(qc.find_bit(instruction[1][0])[0] * 2, ins.copy())
            else:
                if instruction[0].num_qubits > 2:
                    # only 2 qubit gates
                    raise ValueError("Gates with number of Qubits higher than 2 not implemented")
                c_idx = qc.find_bit(instruction[1][0])[0] * 2
                c_data = qc.find_bit(instruction[1][1])[0] * 2
                c_first = min(c_idx, c_data)

                self._create_2_qubit_gates_from_catalog(instruction[0].name, n_cnot, c_idx, c_data, c_first,
                                                           use_postselection)
        self.apply_input_state()
        return self._converted_processor
