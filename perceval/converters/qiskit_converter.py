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

from perceval.components import Processor
from .abstract_converter import AGateConverter


class QiskitConverter(AGateConverter):
    r"""Qiskit quantum circuit to perceval processor converter.

    :param catalog: a component library to use for the conversion. It must contain CNOT gates.
    :param backend_name: backend name used in the converted processor (default SLOS)
    :param source: the source used as input for the converted processor (default perfect source).
    """
    def __init__(self, catalog, **kwargs):
        super().__init__(catalog, **kwargs)

    @property
    def name(self) -> str:
        return "QiskitConverter"

    def set_num_qbits(self, gate_circuit) -> int:
        return gate_circuit.qregs[0].size  # number of Qbits

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

        if self._converted_processor is None:
            self.configure_processor(qc)
        p = self._converted_processor  # empty processor with ports initialized

        for instruction in qc.data:
            # barrier has no effect
            if isinstance(instruction[0], qiskit.circuit.barrier.Barrier):
                continue
            # some limitation in the conversion, in particular measure
            assert isinstance(instruction[0], qiskit.circuit.gate.Gate), "cannot convert (%s)" % instruction[0]

            if instruction[0].num_qubits == 1:
                # one mode gate
                ins = super()._create_generic_1_qubit_gate(instruction[0].to_matrix())
                ins._name = instruction[0].name
                p.add(instruction[1][0].index * 2, ins.copy())
            else:
                c_idx = instruction[1][0].index * 2
                c_data = instruction[1][1].index * 2
                c_first = min(c_idx, c_data)

                p = super()._create_2_qubits_from_catalog(instruction[0].name, n_cnot, cnot_idx, c_idx, c_data, c_first,
                                                          use_postselection)
        return p
