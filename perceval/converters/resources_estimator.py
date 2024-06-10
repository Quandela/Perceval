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

from perceval.converters.circuit_to_graph_converter import CircuitToGraphConverter, gates_and_qubits
import numpy as np
from qiskit import QuantumCircuit


class ResourcesEstimator:
    """
       Estimate the resources required for a given Qiskit Quantum Circuit.

       :param qiskit_circuit: Quantum circuit to estimate resources for.
       :type qiskit_circuit: QuantumCircuit
       :param encoding: Custom encoding to use.
       :type encoding: Optional[List[List[int]]]
       """
    def __init__(self, qiskit_circuit: QuantumCircuit, encoding: list[list[int]] = None):
        self.circuit = qiskit_circuit
        self.gates, self.qubits = gates_and_qubits(self.circuit)
        if encoding is None:
            partitioner = CircuitToGraphConverter(self.circuit)
            partitions, cnots = partitioner.graph_k_clustering_and_cnots_needed(compute_with_min_cnots=True)
            self.encoding = partitions[np.argmin(cnots)]
            # In most cases this will obviously be dual rail. Try to see if you can do it recursively for many of them
        else:
            self.encoding = encoding
        self.needed_entangling_gates, self.needed_photons, self.needed_modes = self.resources()

    def check_same_subset(self):
        same_subset_list = []
        for gate_qubits_sublist in self.qubits:
            same_subset = any(all(qubit in subset for qubit in gate_qubits_sublist) for subset in self.encoding)
            same_subset_list.append(same_subset)
        return same_subset_list

    def resources(self) -> tuple[int, int, int]:
        '''''
        :return: num_cnots, num_photons, num_modes needed to simulate the circuit with the specific encoding
        :rtype:  tuple[int, int, int]
        '''''
        partition = self.encoding
        bool_list = self.check_same_subset()
        false_indices = [index for index, value in enumerate(bool_list) if not value]
        gates_diff_qudits = [self.gates[i] for i in false_indices]
        qubits_diff_qudits = [self.qubits[i] for i in false_indices]

        K = []
        for i in range(len(qubits_diff_qudits)):
            subpartition_lengths = []
            for qubit in qubits_diff_qudits[i]:
                for subpartition in self.encoding:
                    if qubit in subpartition:
                        subpartition_lengths.append(len(subpartition))
                        break
            a, b = subpartition_lengths
            k = 2 ** (a + b - 2)
            if gates_diff_qudits[i] in ['cx', 'cy', 'cz', 'ch', 'cp']:
                K.append(k)
            elif gates_diff_qudits[i] in ['swap', 'iswap']:
                K.append(0)
            else:
                K.append(2 * k)
        num_cnots = sum(K)
        num_photons = len(self.encoding) + 2*sum(K)  # Assuming all CX are heralded
        num_modes = sum(2 ** len(sublist) for sublist in self.encoding)
        return num_cnots, num_photons, num_modes
