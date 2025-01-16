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

from perceval.components import Processor, Source, catalog
from perceval.utils.logging import get_logger, channel
from .abstract_converter import AGateConverter
from .converter_utils import label_cnots_in_gate_sequence
from .circuit_to_graph_converter import gates_and_qubits
from perceval.utils import NoiseModel

def _get_gate_sequence(qisk_circ) -> list:
    # returns a nested list of gate names with corresponding qubit positions
    gate_names, qubit_pos = gates_and_qubits(qisk_circ)  # from qiskit circuit

    gate_info = [[gate_name, q_pos] for gate_name, q_pos in zip(gate_names, qubit_pos)]
    return gate_info


class QiskitConverter(AGateConverter):
    r"""Qiskit quantum circuit to perceval processor converter.

    :param backend_name: backend name used in the converted processor (default SLOS)
    :param source: the source used as input for the converted processor (default perfect source).
    """
    def __init__(self, backend_name: str = "SLOS", source: Source = None, noise_model: NoiseModel = None):
        super().__init__(backend_name, source, noise_model)

    def count_qubits(self, gate_circuit) -> int:
        return gate_circuit.qregs[0].size  # number of qubits

    def convert(self, qc, use_postselection: bool = True) -> Processor:
        r"""Convert a qiskit quantum circuit into a `Processor`.

        :param qc: quantum-based qiskit circuit
        :type qc: qiskit.QuantumCircuit
        :param use_postselection: when True (default), uses optimized number of `postprocessed CNOT` and
            'Heralded CNOT' gates. Otherwise, uses only `heralded CNOT`.

        :return: the converted processor
        """
        import qiskit  # this nested import fixes automatic class reference generation

        get_logger().info(f"Convert qiskit.QuantumCircuit ({qc.num_qubits} qubits, {len(qc.data)} operations) to processor",
                    channel.general)

        gate_sequence = _get_gate_sequence(qc)
        optimized_gate_sequence = label_cnots_in_gate_sequence(gate_sequence)

        qubit_names = qc.qregs[0].name
        self._configure_processor(qc, qname=qubit_names)  # empty processor with ports initialized

        for gate_index, instruction in enumerate(qc.data):
            # barrier has no effect
            if isinstance(instruction.operation, qiskit.circuit.barrier.Barrier):
                continue
            # some limitation in the conversion, in particular measure
            assert isinstance(instruction.operation, qiskit.circuit.gate.Gate), "cannot convert (%s)" % instruction[0]

            gate_name = instruction.operation.name
            if instruction.operation.num_qubits == 1:
                if gate_name == 'p':
                    # not the same name in our catalog
                    gate_name = 'ph'
                elif gate_name == 'sdg':
                    gate_name = 'sdag'
                elif gate_name == 'tdg':
                    gate_name = 'tdag'

                if gate_name in catalog:
                    gate_param = instruction.operation.params[0] if instruction.operation.params else None
                    ins = self._create_catalog_1_qubit_gate(gate_name, param=gate_param)
                else:
                    ins = self._create_generic_1_qubit_gate(instruction.operation.to_matrix())
                    ins._name = instruction.operation.name

                self._converted_processor.add(qc.find_bit(instruction.qubits[0])[0] * 2, ins.copy())
            else:
                if instruction.operation.num_qubits > 2:
                    # only 2 qubit gates
                    raise NotImplementedError("2+ Qubit gates not implemented")
                c_idx = qc.find_bit(instruction.qubits[0])[0] * 2
                c_data = qc.find_bit(instruction.qubits[1])[0] * 2
                self._create_2_qubit_gates_from_catalog(optimized_gate_sequence[gate_index], c_idx, c_data,
                                                        use_postselection)
        self.apply_input_state()
        return self._converted_processor
