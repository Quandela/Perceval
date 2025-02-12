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
from perceval.utils import NoiseModel


class QiskitConverter(AGateConverter):
    r"""Qiskit quantum circuit to perceval processor converter.

    :param backend_name: backend name used in the converted processor (default SLOS)
    :param source: the source used as input for the converted processor (default perfect source).
    """
    def __init__(self, backend_name: str = "SLOS", source: Source = None, noise_model: NoiseModel = None):
        super().__init__(backend_name, source, noise_model)
        import qiskit as qiskit # this nested import fixes automatic class reference generation
        self._qiskit = qiskit

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
        # import qiskit  # this nested import fixes automatic class reference generation

        get_logger().info(f"Convert qiskit.QuantumCircuit ({qc.num_qubits} qubits, {len(qc.data)} operations) to processor",
                    channel.general)

        # some limitation in the conversion, in particular measure
        assert all(isinstance(instruction.operation, self._qiskit.circuit.gate.Gate)
                   for _, instruction in enumerate(qc.data)), \
            "Cannot convert instruction(s): " + ", ".join(
                f"{type(instruction.operation)}" for _, instruction in enumerate(qc.data)
                if not isinstance(instruction.operation, self._qiskit.circuit.gate.Gate))

        qubit_names = qc.qregs[0].name
        self._configure_processor(qc, qname=qubit_names)  # empty processor with ports initialized

        gate_sequence = self._get_gate_sequence(qc)
        return self._generate_converted_processor(gate_sequence, use_postselection=use_postselection)

    @staticmethod
    def _map_gate_names(gate_name: str) -> str:
        # updates gate names to be consistent with Perceval catalog
        if gate_name == 'p':
            return 'ph'
        elif gate_name == 'sdg':
            return 'sdag'
        elif gate_name == 'tdg':
            return 'tdag'
        else:
            return gate_name

    def _get_gate_sequence(self, qisk_circ) -> list[list]:
        """
        Iterates over a Qiskit Circuit to create a list of gate sequence where each element provides
        the necessary gate information for the converter. Each element is a list [gate name, gate position, parameter]

        :param qisk_circ: A qiskit circuit
        :return: A list of gate sequences with names, their positions, parameter (if any).
        """
        gate_sequence = []
        for instruction in qisk_circ.data:
            if isinstance(instruction.operation, self._qiskit.circuit.barrier.Barrier):
                continue

            gate_name = QiskitConverter._map_gate_names(instruction.operation.name)
            qubit_pos = [qisk_circ.find_bit(q).index for q in instruction.qubits]

            need_unitary = False
            if gate_name not in catalog and len(qubit_pos) == 1:
                # use gate unitary to generate any random ! qubit gate
                need_unitary = True

            gate_sequence.append([gate_name, qubit_pos,
                                  instruction.operation.params[0] if instruction.operation.params else None,
                                  instruction.operation.to_matrix() if need_unitary else None])
        return gate_sequence
